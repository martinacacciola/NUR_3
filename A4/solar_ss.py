from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

# Pick a time
t = Time("2021-12-07 10:00")

# Function to initialize positions and velocities
def initialize_positions_velocities(t):
    with solar_system_ephemeris.set('jpl'):
        planets = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
        positions = {}
        velocities = {}
        for planet in planets:
            posvel = get_body_barycentric_posvel(planet, t)
            positions[planet] = posvel[0].xyz.to(u.m).value  # positions in meters
            velocities[planet] = posvel[1].xyz.to(u.m/u.s).value  # velocities in meters per second
    return positions, velocities

# Initialize positions and velocities
initial_positions, initial_velocities = initialize_positions_velocities(t)

## a)
# Extract x, y, z positions
x = np.array([initial_positions[planet][0] for planet in initial_positions]) / u.au.to(u.m)
y = np.array([initial_positions[planet][1] for planet in initial_positions]) / u.au.to(u.m)
z = np.array([initial_positions[planet][2] for planet in initial_positions]) / u.au.to(u.m)

# Define planet names
names = np.array(['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'])

# Plotting (x, y) positions
fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for i, obj in enumerate(names):
    ax[0].scatter(x[i], y[i], label=obj)
    ax[1].scatter(x[i], z[i], label=obj)

ax[0].set_aspect('equal', 'box')
ax[1].set_aspect('equal', 'box')
ax[0].set(xlabel='X [AU]', ylabel='Y [AU]')
ax[1].set(xlabel='X [AU]', ylabel='Z [AU]')

plt.legend(loc=(1.05, 0))
plt.savefig("./plots/fig1a.png")
plt.close()

## b)
# Define necessary parameters and constants
G = 6.67430e-11 * (u.m**3 / (u.kg * u.s**2))  # gravitational constant
m_sun = 1.989e30 * u.kg  # mass of the Sun

# Mass of the planets (in kg)
m_planet = {
    'mercury': 3.3011e23 * u.kg,
    'venus': 4.8675e24 * u.kg,
    'earth': 5.97237e24 * u.kg,
    'mars': 6.4171e23 * u.kg,
    'jupiter': 1.8982e27 * u.kg,
    'saturn': 5.6834e26 * u.kg,
    'uranus': 8.6810e25 * u.kg,
    'neptune': 1.02413e26 * u.kg
}

# Function to compute acceleration
def acceleration(r):
    r_mag = np.sqrt(np.sum(r**2, axis=0))
    a = -G.value * m_sun.value * r / (r_mag**3)
    return a

# Simulation parameters
time_steps = 200 * 365 * 2  # 200 years with a time step of 0.5 days
dt = 0.5 * u.d.to(u.s)  # time step in seconds

# Initialize positions and velocities for Leapfrog integration
positions_lf = {planet: initial_positions[planet].copy() for planet in initial_positions}
velocities_lf = {planet: initial_velocities[planet].copy() for planet in initial_positions}

# Pre-allocate arrays for positions history (Leapfrog)
positions_history_lf = {planet: np.zeros((time_steps, 3)) for planet in initial_positions}

# Run Leapfrog integration
for step in range(time_steps):
    for planet in m_planet:
        # Relative position vector between each planet and the Sun
        r = positions_lf[planet] - positions_lf['sun']
        # Update velocity to the midpoint using the acceleration at the current position
        velocities_lf[planet] += acceleration(r) * (dt / 2)
        # Update positions using the midpoint velocity
        positions_lf[planet] += velocities_lf[planet] * dt
        # Store current position in AU
        positions_history_lf[planet][step] = positions_lf[planet] / u.au.to(u.m)
    # Update velocities again using the new position
    for planet in m_planet:
        r = positions_lf[planet] - positions_lf['sun']
        velocities_lf[planet] += acceleration(r) * (dt / 2)

# Extract x, y, z coordinates for all planets in AU (Leapfrog)
x_leapfrog = [positions_history_lf[planet][:, 0] for planet in initial_positions]
y_leapfrog = [positions_history_lf[planet][:, 1] for planet in initial_positions]
z_leapfrog = [positions_history_lf[planet][:, 2] for planet in initial_positions]

# Convert time to years
time = np.linspace(0, 200, time_steps)

# Main plot with two subplots for all planets (Leapfrog)
fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
# Plot x-y plane for all planets
for i, planet in enumerate(initial_positions):
    ax[0].plot(x_leapfrog[i], y_leapfrog[i], label=planet)
# Plot time versus z plane for all planets
for i, planet in enumerate(initial_positions):
    ax[1].plot(time, z_leapfrog[i], label=planet)
ax[0].set_aspect('equal', 'box')
ax[0].set(xlabel='X [AU]', ylabel='Y [AU]')
ax[1].set(xlabel='Time [yr]', ylabel='Z [AU]')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.savefig("./plots/fig1b.png")
plt.close()

# Zoom on the inner planets (Leapfrog)
fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
inner_planets = ['sun', 'mercury', 'venus', 'earth', 'mars']
for i, planet in enumerate(inner_planets):
    ax[0].plot(x_leapfrog[i], y_leapfrog[i], label=planet)
    ax[1].plot(time, z_leapfrog[i], label=planet)
ax[0].set_aspect('equal', 'box')
ax[0].set(xlabel='X [AU]', ylabel='Y [AU]')
ax[1].set(xlabel='Time [yr]', ylabel='Z [AU]')
ax[1].set_xlim(0, 5)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.savefig("./plots/fig1b_zoom.png")
plt.close()

## c)
# Runge-Kutta integration function
def runge_kutta_step(r, v, dt):
    # Initial velocity and position
    k1v = acceleration(r)
    k1r = v
    # Update velocity and position by weighted combination of slopes
    k2v = acceleration(r + k1r * (dt / 2))
    k2r = v + k1v * (dt / 2)
    k3v = acceleration(r + k2r * (dt / 2))
    k3r = v + k2v * (dt / 2)
    k4v = acceleration(r + k3r * dt)
    k4r = v + k3v * dt
    # Return new position and velocity
    v_new = v + (1 / 6) * (k1v + 2*k2v + 2*k3v + k4v) * dt
    r_new = r + (1 / 6) * (k1r + 2*k2r + 2*k3r + k4r) * dt
    return r_new, v_new

# Initialize positions and velocities for Runge-Kutta integration
positions_rk = {planet: initial_positions[planet].copy() for planet in initial_positions}
velocities_rk = {planet: initial_velocities[planet].copy() for planet in initial_positions}

# Pre-allocate arrays for positions history (Runge-Kutta)
positions_history_rk = {planet: np.zeros((time_steps, 3)) for planet in initial_positions}

# Run Runge-Kutta integration
for step in range(time_steps):
    # Update positions and velocities for all planets
    for planet in m_planet:
        positions_rk[planet], velocities_rk[planet] = runge_kutta_step(positions_rk[planet], velocities_rk[planet], dt)
        # Store current position in AU
        positions_history_rk[planet][step] = positions_rk[planet] / u.au.to(u.m)

# Extract x, y, z coordinates for all planets in AU (Runge-Kutta)
x_rk = [positions_history_rk[planet][:, 0] for planet in initial_positions]
y_rk = [positions_history_rk[planet][:, 1] for planet in initial_positions]
z_rk = [positions_history_rk[planet][:, 2] for planet in initial_positions]

# Main plot with two subplots for all planets (Runge-Kutta)
fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
# Plot x-y plane for all planets
for i, planet in enumerate(initial_positions):
    ax[0].plot(x_rk[i], y_rk[i], label=planet)
# Plot time versus z plane for all planets
for i, planet in enumerate(initial_positions):
    ax[1].plot(time, z_rk[i], label=planet)
ax[0].set_aspect('equal', 'box')
ax[0].set(xlabel='X [AU]', ylabel='Y [AU]')
ax[1].set(xlabel='Time [yr]', ylabel='Z [AU]')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.savefig("./plots/fig1c.png")
plt.close()

# Zoom on the inner planets (Runge-Kutta)
fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for i, planet in enumerate(inner_planets):
    ax[0].plot(x_rk[i], y_rk[i], label=planet)
    ax[1].plot(time, z_rk[i], label=planet)
ax[0].set_aspect('equal', 'box')
ax[0].set(xlabel='X [AU]', ylabel='Y [AU]')
ax[1].set(xlabel='Time [yr]', ylabel='Z [AU]')
ax[1].set_xlim(0, 5)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.savefig("./plots/fig1c_zoom.png")
plt.close()

# Calculate difference in x positions between Leapfrog and Runge-Kutta methods
x_diff = np.abs(np.array(x_leapfrog) - np.array(x_rk))

# Create a plot of the difference in x positions versus time
plt.figure(figsize=(8, 6))
for i, planet in enumerate(m_planet):
    plt.plot(time, x_diff[i], label=planet, alpha=0.5)
plt.xlabel('Time [yr]')
plt.ylabel('Difference in X Position [AU]')
plt.title('Difference in X Position Between Leapfrog and Runge-Kutta')
plt.legend()
plt.savefig("./plots/fig1c_diff.png")
plt.close()



