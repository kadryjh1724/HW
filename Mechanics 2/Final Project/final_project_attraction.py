import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist, squareform
from datetime import datetime

# Number of particles
N = int(input('Number of particles : '))
# Time
time = 5
# Frames per second
FPS = 60
# Box size
B = 10.0
# Temperature
T = float(input('Temperature : '))
# Lennard-Jones Parameter
sigma = 0.1
eps = 1e-16
# Cutoff distance for LJ potential
c = 1.0

# Initial Conditions
# ic_r = np.random.rand(N,2) * B
ic_r = np.random.rand(N,2) * B / 2
ic_v = np.random.randn(N,2) * T
dt = 1.0 / FPS

# Setting figures
fig = plt.figure()
ax = fig.add_subplot(111, aspect = 'equal', autoscale_on = False, xlim = (0,B), ylim = (0,B))
particles, = ax.plot([], [], 'bo', ms = 6)

# Initial plot
def init():
    global particles
    particles.set_data(ic_r[:,0], ic_r[:,1])
    return particles,

def animate(i):
    global ic_r, ic_v, particles

    # Find interacting particles
    dist = squareform(pdist(ic_r))
    I1, I2 = np.where(dist < c)
    unique = (I1 < I2)
    I1 = I1[unique]
    I2 = I2[unique]

    # Scaled distance
    X = dist / sigma
    for j in range(0,N):
        for k in range(0,N):
            if X[j,k] <= 0.2:
                X[j,k] = 0.2
    # Force
    F = 24 * eps * np.power(X,-7) * (2 * np.power(X,-6) - 1) / sigma

    for i1, i2 in zip(I1, I2):
        r1 = ic_r[i1,]
        r2 = ic_r[i2,]
        v1 = ic_v[i1,]
        v2 = ic_v[i2,]

        # Relative position and force
        r12 = r1 - r2
        f = F[i1,i2]
        f1 = f * r12
        f2 = -f * r12

        # Update velocities
        ic_v[i1,:] = v1 + f1 * dt 
        ic_v[i2,:] = v2 + f2 * dt
        
    ic_r += dt * ic_v
    ic_r = ic_r % B

    particles.set_data(ic_r[:,0], ic_r[:,1])
    return particles,

ani = animation.FuncAnimation(fig, animate, frames = time * FPS, interval = time, blit = True, init_func = init)
ani.save('LJ_{0:}_ptcls_{1:}_T_{2:}.mp4'.format(N,T,datetime.now().strftime('%d_%b_%Y_%H:%M:%S')), fps = 60, extra_args = ['-vcodec', 'libx264'])




















