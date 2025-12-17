import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Create data
t = np.linspace(0, 2*np.pi, 200)
x = np.cos(t)
y = np.sin(t)
z = t

# Create figure and 3D axis
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Set isometric-like view
ax.view_init(elev=30, azim=45)

# Axis limits
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(0, 2*np.pi)

line, = ax.plot([], [], [], lw=2)

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

def update(frame):
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])
    return line,

ani = FuncAnimation(
    fig,
    update,
    frames=len(t),
    init_func=init,
    blit=False  # IMPORTANT for 3D
)

# Save animation
writer = FFMpegWriter(fps=30, bitrate=1800)
ani.save("3d_isometric_animation.mp4", writer=writer)

plt.close()
