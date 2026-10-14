## Kaptiza Pendulum Proposal

# Background information 
We first begin with a simple pendulum. The equation of motion may be obtained through the Lagrangian. The equilibium positions $\theta_e=0,\pi$ can be solved for by setting 
$\dot \theta = 0$ and solving for $\theta$. Small perturbations around equilibrium can be analyzed by setting $\theta= \delta + \theta_e$ and linearizing the equations. Solving the resultant ODE's we find that $\theta_e=0$ is a point of stable equilibrium and $\theta_e=\pi$ is unstable equilibrium.
\
For Kapitza's pendulum we follow the same model except there is an added $y_p = a\cos(\omega t)$ for the y-displacememnt of the pendulum due to the oscillating pivot, where $\omega$ is the driving frequency of the pivot and $a$ is the amplitude.
After perturbing about equilibrium and linearizing we find that 
$$
\ddot \delta + [1/2(\frac{a\omega}{l})^2 \pm \omega_0^2]\delta = 0
$$
where $\omega_0$ is the natural frequency of the pendulum, and the $\pm$ stands for $\theta_e=0$ and $\theta_e=\pi$, respectively. In particular, if the driving frequency $\omega^{-1}, a << 1$  and $\frac{a\omega}{l} \sim \omega_0$ we have dynamic stable equilibrium about $\theta_e = 0, \pi$.




# Background information
    - ODE's stability theory : regular pendulum, unstable equilibrium (inverted pozition)
    - ODE's stability thoery : kapitza pendulum, stable equilibrium  (inverted position)

    - why does it happen? FBD , lagrangian, eigenvalues etc...
# Research Variables
    - motor speed
    - orbit 
    - noise spectrum of driving force (fourrier series)
    - length / mass distribution (moment inertia)
    - Initial conditions (angle, velocity)
    - double, triple pendulum ?
# Experimental setup
    - bearings
    - makerspace: plastic pendulum
    - arduino / stepper motor
    - stable table
    - C-clamps
    - rod