# import numpy as np
# from matplotlib import pyplot as plt, animation


# simulation_size = 200
# Ex = np.zeros(simulation_size)
# Hy = np.zeros(simulation_size)

# dx = 0.01
# dt = dx / 6e8
# freq_in = 7e8
# timesteps = 1000

# j_source = 5

# medium_start = 100
# medium_end = medium_start + 50
# eps0 = 8.85e-12
# eps = np.ones(simulation_size) * 0.5
# conductivity_correction = np.ones(simulation_size)
# eps_r = 4
# sigma = 0.05
# conductivity_parameter = dt * sigma / (2 * eps_r * eps0)

# eps[medium_start:medium_end] = 0.5 / eps_r
# conductivity_correction[medium_start:medium_end] = (1 - conductivity_parameter) / (
#     1 + conductivity_parameter
# )

# boundary_low = [0, 0]
# boundary_high = [0, 0]

# E_frames = []
# H_frames = []

# for timestep in range(1, timesteps + 1):
#     Ex[:-1] = conductivity_correction[1:] * Ex[:-1] + eps[1:] * (Hy[:-1] - Hy[1:])

#     pulse = np.sin(2 * np.pi * freq_in * dt * timestep)
#     Ex[j_source] += pulse

#     Ex[0] = boundary_low.pop(0)
#     boundary_low.append(Ex[1])
#     Ex[-1] = boundary_high.pop(0)
#     boundary_high.append(Ex[-2])

#     Hy[1:] = Hy[1:] + 0.5 * (Ex[:-1] - Ex[1:])

#     E_frames.append(Ex.copy())
#     H_frames.append(Hy.copy())

# frames = []
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# for frame in range(timesteps):
#     (im,) = ax.plot(frame, color="blue")
#     frames.append([im])

# ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
# plt.show()


# import numpy as np
# from matplotlib import pyplot as plt, animation


# simulation_size = 200
# Ex = np.zeros(simulation_size)
# Hy = np.zeros(simulation_size)

# dx = 0.01
# dt = dx / 6e8
# freq_in = 7e8
# timesteps = 1000  # shorter for demo

# j_source = 5

# medium_start = 100
# medium_end = medium_start + 50
# eps0 = 8.85e-12
# eps = np.ones(simulation_size) * 0.5
# conductivity_correction = np.ones(simulation_size)
# eps_r = 4
# sigma = 0.05
# conductivity_parameter = dt * sigma / (2 * eps_r * eps0)

# eps[medium_start:medium_end] = 0.5 / eps_r
# conductivity_correction[medium_start:medium_end] = (1 - conductivity_parameter) / (
#     1 + conductivity_parameter
# )

# boundary_low = [0, 0]
# boundary_high = [0, 0]

# E_frames = []
# H_frames = []

# for timestep in range(1, timesteps + 1):
#     Ex[:-1] = conductivity_correction[1:] * Ex[:-1] + eps[1:] * (Hy[:-1] - Hy[1:])

#     pulse = np.sin(2 * np.pi * freq_in * dt * timestep)
#     Ex[j_source] += pulse

#     Ex[0] = boundary_low.pop(0)
#     boundary_low.append(Ex[1])
#     Ex[-1] = boundary_high.pop(0)
#     boundary_high.append(Ex[-2])

#     Hy[1:] = Hy[1:] + 0.5 * (Ex[:-1] - Ex[1:])

#     E_frames.append(Ex.copy())
#     H_frames.append(Hy.copy())

# # ------------------ ANIMATION ------------------

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")

# x = np.arange(simulation_size) * dx  # spatial axis

# artists = []

# for t in range(timesteps):
#     # Plot E along Y-axis
#     (lineE,) = ax.plot(
#         x,
#         E_frames[t],
#         np.zeros_like(x),
#         color="blue",
#         label="E-field" if t == 0 else "",
#     )

#     # Plot H along Z-axis (perpendicular)
#     (lineH,) = ax.plot(
#         x, np.zeros_like(x), H_frames[t], color="red", label="H-field" if t == 0 else ""
#     )

#     artists.append([lineE, lineH])

# ani = animation.ArtistAnimation(fig, artists, interval=20, blit=True, repeat_delay=1000)

# # Set labels
# ax.set_xlabel("x")
# ax.set_ylabel("y (E field)")
# ax.set_zlabel("z (H field)")
# ax.set_title("E and H Field Animation")

# ax.legend()

# plt.show()


import numpy as np
from matplotlib import pyplot as plt, animation

# ------------------ FDTD SETUP ------------------

simulation_size = 200
Ex = np.zeros(simulation_size)
Hy = np.zeros(simulation_size)

dx = 0.01
dt = dx / 6e8
freq_in = 7e8
timesteps = 1000  # fewer timesteps for demo

j_source = 5

medium_start = 100
medium_end = medium_start + 50
eps0 = 8.85e-12
eps = np.ones(simulation_size) * 0.5
conductivity_correction = np.ones(simulation_size)
eps_r = 4
sigma = 0.99
conductivity_parameter = dt * sigma / (2 * eps_r * eps0)

eps[medium_start:medium_end] = 0.5 / eps_r
conductivity_correction[medium_start:medium_end] = (1 - conductivity_parameter) / (
    1 + conductivity_parameter
)

boundary_low = [0, 0]
boundary_high = [0, 0]

E_frames = []
H_frames = []

for timestep in range(1, timesteps + 1):
    Ex[:-1] = conductivity_correction[1:] * Ex[:-1] + eps[1:] * (Hy[:-1] - Hy[1:])

    pulse = np.sin(2 * np.pi * freq_in * dt * timestep)
    Ex[j_source] += pulse

    Ex[0] = boundary_low.pop(0)
    boundary_low.append(Ex[1])
    Ex[-1] = boundary_high.pop(0)
    boundary_high.append(Ex[-2])

    Hy[1:] = Hy[1:] + 0.5 * (Ex[:-1] - Ex[1:])

    E_frames.append(Ex.copy())
    H_frames.append(Hy.copy())

# ------------------ ANIMATION ------------------

fig = plt.figure(figsize=(18, 6))

# Left: E-field (2D)
axE = fig.add_subplot(131)
# Middle: H-field (2D)
axH = fig.add_subplot(132)
# Right: Both (3D)
axBoth = fig.add_subplot(133, projection="3d")

x = np.arange(simulation_size)  # spatial axis

artists = []

for t in range(timesteps):
    frame_artists = []

    # ---- Left subplot: E only (2D) ----
    (lineE2D,) = axE.plot(x, E_frames[t], color="red")
    frame_artists.append(lineE2D)

    # ---- Middle subplot: H only (2D) ----
    (lineH2D,) = axH.plot(x, H_frames[t], color="blue")
    frame_artists.append(lineH2D)

    # ---- Right subplot: Both (3D) ----
    (lineE3D,) = axBoth.plot(
        x,
        np.zeros_like(x),
        E_frames[t],
        color="red",
        label="E-field" if t == 0 else "",
    )
    (lineH3D,) = axBoth.plot(
        x,
        H_frames[t],
        np.zeros_like(x),
        color="blue",
        label="H-field" if t == 0 else "",
    )
    frame_artists.extend([lineE3D, lineH3D])

    artists.append(frame_artists)

ani = animation.ArtistAnimation(fig, artists, interval=30, blit=True, repeat_delay=1000)

# Labels & titles
axE.set_xlabel("z")
axE.set_ylabel("Ex")
axE.set_title("Electric field (Ex)")

axH.set_xlabel("z")
axH.set_ylabel("Hy")
axH.set_title("Magnetic field (Hy)")

axBoth.set_xlabel("z")
axBoth.set_ylabel("y")
axBoth.set_zlabel("x")
axBoth.set_title("Ex & Hy fields")
axBoth.legend()

# plt.tight_layout()
plt.show()
