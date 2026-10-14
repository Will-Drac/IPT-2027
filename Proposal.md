## Kaptiza Pendulum Proposal

# Background information 
We first begin with a simple pendulum. The equation of motion may be obtained through the Lagrangian. The equilibrium positions $\theta_e=0,\pi$ can be solved for by setting 
$\dot \theta = 0$ and solving for $\theta$. Small perturbations around equilibrium can be analyzed by setting $\theta= \delta + \theta_e$ and linearizing the equations. Solving the resultant ODE's we find that $\theta_e=0$ is a point of stable equilibrium and $\theta_e=\pi$ is unstable equilibrium.
\
For Kapitza's pendulum we follow the same model except there is an added $y_p = a\cos(\omega t)$ for the y-displacement of the pendulum due to the oscillating pivot, where $\omega$ is the driving frequency of the pivot and $a$ is the amplitude.
After perturbing about equilibrium and linearizing we find that 
$$
\ddot \delta + [1/2(\frac{a\omega}{l})^2 \pm \omega_0^2]\delta = 0
$$
where $\omega_0$ is the natural frequency of the pendulum, and the $\pm$ stands for $\theta_e=0$ and $\theta_e=\pi$, respectively. In particular, if the driving frequency $\omega^{-1}, a << 1$  and $\frac{a\omega}{l} \sim \omega_0$ we have dynamic stable equilibrium about $\theta_e = 0, \pi$.

# Research variables
The Project sets out to confirm the theory. In particular, varying the parameters $a, \omega, l$ (drive amplitude, drive frequency, length of pendulum) and confirming the conditions under which they guarantee dynamic stability. Initial conditions (angle, velocity) may be varied and the orbits which follow compared against the expected phase diagrams. The pendulum mass and in particular mass distribution may be varied.


Then, the pivot's oscillation will be superimposed or replaced by noise and the resulting stability state will be analyzed experimentally and theoretically as a function of the noise spectrum. Time permitting, double (even triple) Kapitza pendulum's may be experimented with. 


# Experimental setup
- bearings
- makerspace: plastic pendulum
- arduino / stepper motor
- stable table
- C-clamps
- rod