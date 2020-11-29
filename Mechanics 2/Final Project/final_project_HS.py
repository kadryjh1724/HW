import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist, squareform
from datetime import datetime


# Number of particles
N = int(input('Number of particles : '))
# Time
time = 10
# Frames per second
FPS = 60
# Box size
B = 5.0
# Temperature
T = float(input('Temperature : '))

# Initial Conditions
# ic_r = np.random.rand(N,2) * B
ic_r = np.random.rand(N,2) * B / 2
ic_v = np.random.randn(N,2) * T
dt = 1.0 / FPS

dist = squareform(pdist(ic_r))
print('Interparticle distance: mean ',np.mean(dist),' stdev ',np.std(dist))

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

    dist = squareform(pdist(ic_r))
    I1, I2 = np.where(dist < 0.3)
    unique = (I1 < I2)
    I1 = I1[unique]
    I2 = I2[unique]

    for i1, i2 in zip(I1, I2):
        r1 = ic_r[i1,]
        r2 = ic_r[i2,]
        v1 = ic_v[i1,]
        v2 = ic_v[i2,]

        # Relative position and velocity
        r12 = r1 - r2
        v12 = v1 - v2

        # CM velocity
        v_cm = (v1 + v2) / 2

        r_rel = np.dot(r12, r12)
        v_rel = np.dot(v12, r12)
        v_12 = 2 * r12 * v_rel / r_rel - v12
        
        ic_v[i1,:] = v_cm + v_12/2
        ic_v[i2,:] = v_cm - v_12/2
    ic_r += dt * ic_v
    ic_r = ic_r % B

    particles.set_data(ic_r[:,0], ic_r[:,1])
    return particles,

ani = animation.FuncAnimation(fig, animate, frames = time * FPS, interval = time, blit = True, init_func = init)
ani.save('HS_{0:}_ptcls_{1:}_T_{2:}.mp4'.format(N,T,datetime.now().strftime('%d_%b_%Y_%H:%M:%S')), fps = 60, extra_args = ['-vcodec', 'libx264'])
dist = squareform(pdist(ic_r))
print('Interparticle distance after simulation: mean ',np.mean(dist),' stdev ',np.std(dist))



















