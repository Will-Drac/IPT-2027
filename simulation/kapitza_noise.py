"""
Many Kapitza pendulums, driven by random noise, simulated in parallel with NVIDIA Warp.

Global settings (same for every pendulum): noise type / power / frequency band,
natural frequency w0, gravity g, initial angle, initial angular velocity.
Per-pendulum randomness: the amplitude and phase of every noise sinusoid.

Equation of motion (theta = 0 is hanging down, theta = pi is inverted):

    theta'' + w0^2 (1 - p_g'') sin(theta) = 0,        p_g'' = p''/g

where p(t) is the pivot's vertical position, built as a sum of sinusoids:

    p(t) = sum_k a_k cos(W_k t + phi_k)
    p''(t) = -sum_k W_k^2 a_k cos(W_k t + phi_k)

Noise definition (follows noise.ipynb):
  * frequencies W_k = W_min, W_min + dW, ... <= W_max
  * power spectrum P(W) = A W^-alpha, with A fixed by the requested total power P_tot
  * every sinusoid gets a uniformly random phase
  * every sinusoid gets a random amplitude whose MEAN is sqrt(P(W_k) dW)
    (Rayleigh distributed; the dW makes the discrete sum of powers approximate P_tot)
  * the signal is evaluated directly as a sum of sinusoids inside the Warp kernel,
    which is the same signal an inverse FFT on this frequency grid would give,
    without storing a time series for every pendulum.

Usage examples:
    python kapitza_noise.py                       # interactive window
    python kapitza_noise.py --save out.mp4        # or out.gif
    python kapitza_noise.py -n 20000 --alpha 2 --total-power 5e-4
"""

import argparse
import math

import numpy as np
import warp as wp

wp.config.quiet = True

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Noise generation (host side, numpy)
# --------------------------------------------------------------------------- #
def noise_normalisation(alpha, w_min, w_max, total_power):
    """Scale factor A in P(W) = A W^-alpha so that the integral of P over [w_min, w_max] is total_power."""
    if alpha == 1.0:
        return total_power / math.log(w_max / w_min)
    return total_power * (1.0 - alpha) / (w_max ** (1.0 - alpha) - w_min ** (1.0 - alpha))


def make_noise(n_pend, alpha, w_min, w_max, d_w, total_power, g, rng):
    """
    Returns
      freqs  (K,)      float64  noise frequencies W_k
      coef   (N, K)    float32  W_k^2 a_ik / g   (so p_g'' = -sum_k coef cos(W_k t + phase))
      phase  (N, K)    float32  uniform random phases in [0, 2 pi)
      amp    (N, K)    float64  amplitudes a_ik (metres), kept for diagnostics
    """
    n_freq = int(math.floor((w_max - w_min) / d_w + 1e-9)) + 1
    freqs = w_min + d_w * np.arange(n_freq, dtype=np.float64)

    A = noise_normalisation(alpha, w_min, w_max, total_power)
    power = A * freqs ** (-alpha)                 # P(W_k)
    mean_amp = np.sqrt(power * d_w)               # desired mean amplitude of sinusoid k

    # Rayleigh with scale s has mean s*sqrt(pi/2)  ->  s = mean * sqrt(2/pi)
    scale = mean_amp * math.sqrt(2.0 / math.pi)
    amp = rng.rayleigh(scale=scale, size=(n_pend, n_freq))
    phase = rng.uniform(0.0, TWO_PI, size=(n_pend, n_freq))

    coef = (freqs ** 2)[None, :] * amp / g
    return freqs, coef.astype(np.float32), phase.astype(np.float32), amp


def stability_metric(amp, freqs, w0, g):
    """
    Per-pendulum value of  w0^2 * sum_n W_n^2 R_n^2 (with R = a/g, i.e. in p_g units).
    The averaged (Kapitza) theory says the inverted position is stable when this is > 2.
    """
    return w0 ** 2 * np.sum((freqs[None, :] * amp / g) ** 2, axis=1)


# --------------------------------------------------------------------------- #
# Warp kernels
# --------------------------------------------------------------------------- #
@wp.func
def pivot_accel_g(
    t: wp.float64,
    i: int,
    n_freq: int,
    freqs: wp.array(dtype=wp.float64),
    coef: wp.array2d(dtype=wp.float32),
    phase: wp.array2d(dtype=wp.float32),
) -> wp.float32:
    """p_g''(t) for pendulum i. The W*t product is reduced mod 2 pi in float64 so long runs keep their phase accuracy."""
    two_pi = wp.float64(6.283185307179586)
    s = wp.float32(0.0)
    for k in range(n_freq):
        arg = wp.float32(wp.mod(freqs[k] * t, two_pi)) + phase[i, k]
        s += coef[i, k] * wp.cos(arg)
    return -s


@wp.kernel
def advance(
    theta: wp.array(dtype=wp.float32),
    omega: wp.array(dtype=wp.float32),
    freqs: wp.array(dtype=wp.float64),
    coef: wp.array2d(dtype=wp.float32),
    phase: wp.array2d(dtype=wp.float32),
    t0: wp.float64,
    dt: wp.float64,
    n_steps: int,
    n_freq: int,
    w0sq: wp.float32,
):
    """Advance one pendulum n_steps with RK4. One thread = one pendulum."""
    i = wp.tid()
    th = theta[i]
    w = omega[i]
    h = wp.float32(dt)

    for s in range(n_steps):
        t = t0 + wp.float64(s) * dt
        a0 = pivot_accel_g(t, i, n_freq, freqs, coef, phase)
        a1 = pivot_accel_g(t + wp.float64(0.5) * dt, i, n_freq, freqs, coef, phase)
        a2 = pivot_accel_g(t + dt, i, n_freq, freqs, coef, phase)

        k1t = w
        k1w = -w0sq * (1.0 - a0) * wp.sin(th)

        k2t = w + 0.5 * h * k1w
        k2w = -w0sq * (1.0 - a1) * wp.sin(th + 0.5 * h * k1t)

        k3t = w + 0.5 * h * k2w
        k3w = -w0sq * (1.0 - a1) * wp.sin(th + 0.5 * h * k2t)

        k4t = w + h * k3w
        k4w = -w0sq * (1.0 - a2) * wp.sin(th + h * k3t)

        th = th + (h / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t)
        w = w + (h / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)

    theta[i] = th
    omega[i] = w


# --------------------------------------------------------------------------- #
# Simulation wrapper
# --------------------------------------------------------------------------- #
class KapitzaEnsemble:
    def __init__(self, args, device=None):
        self.device = device
        self.n = args.n
        self.w0 = args.w0
        self.g = args.gravity
        rng = np.random.default_rng(args.seed)

        freqs, coef, phase, amp = make_noise(
            self.n, args.alpha, args.w_min, args.w_max, args.d_w, args.total_power, self.g, rng
        )
        self.n_freq = len(freqs)
        self.metric = stability_metric(amp, freqs, self.w0, self.g)

        self.freqs = wp.array(freqs, dtype=wp.float64, device=device)
        self.coef = wp.array(coef, dtype=wp.float32, device=device)
        self.phase = wp.array(phase, dtype=wp.float32, device=device)
        self.theta = wp.array(np.full(self.n, args.theta0, dtype=np.float32), dtype=wp.float32, device=device)
        self.omega = wp.array(np.full(self.n, args.omega0, dtype=np.float32), dtype=wp.float32, device=device)

        self.dt = args.dt if args.dt else 0.1 / args.w_max
        self.t = 0.0
        self.w0sq = self.w0 ** 2

    def step(self, sim_time):
        """Advance every pendulum by (about) sim_time seconds."""
        n_steps = max(1, int(round(sim_time / self.dt)))
        wp.launch(
            advance,
            dim=self.n,
            inputs=[
                self.theta, self.omega, self.freqs, self.coef, self.phase,
                wp.float64(self.t), wp.float64(self.dt), n_steps, self.n_freq, wp.float32(self.w0sq),
            ],
            device=self.device,
        )
        self.t += n_steps * self.dt

    def state(self):
        return self.theta.numpy(), self.omega.numpy()


# --------------------------------------------------------------------------- #
# Animation
# --------------------------------------------------------------------------- #
def wrap_around_pi(theta):
    """Wrap angle into [0, 2 pi) so the inverted position (pi) sits in the middle of the plot."""
    return np.mod(theta, TWO_PI)


def is_upright(theta):
    """True where the pendulum is within pi/2 of the inverted position (above the horizontal)."""
    return np.abs(np.mod(theta, TWO_PI) - math.pi) < math.pi / 2


def run(args):
    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    wp.init()
    sim = KapitzaEnsemble(args, device=args.device)

    print(f"device: {sim.theta.device}")
    print(f"{sim.n} pendulums, {sim.n_freq} noise frequencies in [{args.w_min}, {args.w_max}] rad/s, dt = {sim.dt:.3e} s")
    print(
        f"stability metric w0^2 sum(W^2 R^2)  (stable if > 2): "
        f"mean {sim.metric.mean():.3g}, min {sim.metric.min():.3g}, max {sim.metric.max():.3g}, "
        f"fraction > 2: {np.mean(sim.metric > 2):.2f}"
    )

    frames = args.frames
    per_frame = args.duration / frames

    th, om = sim.state()
    colors = None
    cmap = None
    if args.color_by_theory:
        colors = (sim.metric > 2).astype(float)
        cmap = "coolwarm_r"  # red = predicted unstable, blue = predicted stable

    fig, ax = plt.subplots(figsize=(8, 6))
    ylim = args.ylim if args.ylim else 3.0 * args.w0
    ax.set_xlim(0, TWO_PI)
    ax.set_ylim(-ylim, ylim)
    ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, TWO_PI])
    ax.set_xticklabels(["0\n(down)", r"$\pi/2$", r"$\pi$" + "\n(inverted)", r"$3\pi/2$", r"$2\pi$" + "\n(down)"])
    ax.axvline(math.pi, color="gray", lw=0.8, ls="--")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    style = dict(c=colors, cmap=cmap, vmin=0, vmax=1) if colors is not None else dict(color="tab:blue")
    scat = ax.scatter(wrap_around_pi(th), om, s=args.point_size, alpha=args.alpha_pts, linewidths=0, **style)
    title = ax.set_title("")

    # A pendulum "survives" if it has stayed above the horizontal at every frame so far.
    # (Without damping a fallen pendulum swings back up, so the instantaneous angle alone is misleading.)
    survived = is_upright(th)
    last = {"frame": -1}

    def draw(frame):
        nonlocal survived
        if frame > last["frame"] and frame > 0:   # guard against matplotlib re-drawing a frame
            sim.step(per_frame)
        last["frame"] = max(last["frame"], frame)
        th, om = sim.state()
        survived &= is_upright(th)
        scat.set_offsets(np.column_stack([wrap_around_pi(th), om]))
        title.set_text(
            f"t = {sim.t:6.2f} s   |   {100 * survived.mean():5.1f}% never fell below horizontal"
        )
        return scat, title

    ani = FuncAnimation(fig, draw, frames=frames + 1, interval=1000 / args.fps, blit=False, repeat=False)

    if args.save:
        ani.save(args.save, fps=args.fps, dpi=args.dpi)
        print(f"saved {args.save}")
    else:
        plt.show()

    print(f"final: {100 * survived.mean():.1f}% of pendulums never fell below horizontal by t = {sim.t:.2f} s")
    if args.color_by_theory:
        hi = sim.metric > 2
        if hi.any() and (~hi).any():
            print(f"  survived: {100 * survived[hi].mean():.1f}% of theory-stable, {100 * survived[~hi].mean():.1f}% of theory-unstable")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("global physical settings")
    g.add_argument("--w0", type=float, default=5.0, help="natural frequency of the pendulum, rad/s")
    g.add_argument("--gravity", type=float, default=9.81, help="gravitational acceleration, m/s^2")
    g.add_argument("--theta0", type=float, default=math.pi - 0.2, help="initial angle, rad (pi = inverted)")
    g.add_argument("--omega0", type=float, default=0.0, help="initial angular velocity, rad/s")

    g = p.add_argument_group("noise (applied to the pivot position p, in metres)")
    g.add_argument("--alpha", type=float, default=1.0,
                   help="P(W) ~ W^-alpha: -1 blue, 0 white, 1 pink, 2 brown")
    g.add_argument("--total-power", type=float, default=1e-3, help="P_tot = integral of P(W) over the band (m^2)")
    g.add_argument("--w-min", type=float, default=50.0, help="minimum noise frequency, rad/s (>> w0)")
    g.add_argument("--w-max", type=float, default=200.0, help="maximum noise frequency, rad/s")
    g.add_argument("--d-w", type=float, default=1.0, help="frequency spacing, rad/s (smaller = longer before noise repeats)")

    g = p.add_argument_group("simulation")
    g.add_argument("-n", type=int, default=2000, help="number of pendulums")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--duration", type=float, default=10.0, help="simulated seconds")
    g.add_argument("--frames", type=int, default=300, help="animation frames over the duration")
    g.add_argument("--dt", type=float, default=0.0, help="RK4 step, s (default 0.1 / w_max)")
    g.add_argument("--device", default=None, help='Warp device, e.g. "cuda:0" or "cpu" (default: Warp default)')

    g = p.add_argument_group("display")
    g.add_argument("--fps", type=int, default=30)
    g.add_argument("--ylim", type=float, default=0.0, help="phase plot y half-range (default 3*w0)")
    g.add_argument("--point-size", type=float, default=6.0)
    g.add_argument("--alpha-pts", type=float, default=0.6, help="point opacity")
    g.add_argument("--color-by-theory", action="store_true",
                   help="colour each point by whether the averaged theory predicts its noise to stabilise the inverted state")
    g.add_argument("--save", default="", help="write animation to this file (.mp4 needs ffmpeg, .gif needs pillow)")
    g.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
