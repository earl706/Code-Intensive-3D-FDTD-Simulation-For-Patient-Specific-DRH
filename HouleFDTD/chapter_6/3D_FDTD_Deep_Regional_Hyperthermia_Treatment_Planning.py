"""
fd3d_apa.py: 3D FDTD simulation for Deep Regional Hyperthermia Treatment Planning
Based on Chapter 6 of "Electromagnetic Simulation Using the FDTD Method with Python"
by Jennifer E. Houle and Dennis M. Sullivan (3rd Edition, 2020)
"""

from math import sqrt, sin, cos, pi
import numba
import numpy as np
from matplotlib import pyplot as plt


# Functions for main FDTD loop
def calculate_pml_parameters(
    npml, simulation_size_x, simulation_size_y, simulation_size_z
):
    """Calculate and return the PML parameters"""
    gi1 = np.zeros(simulation_size_x)
    gi2 = np.ones(simulation_size_x)
    gi3 = np.ones(simulation_size_x)
    fi1 = np.zeros(simulation_size_x)
    fi2 = np.ones(simulation_size_x)
    fi3 = np.ones(simulation_size_x)
    gj1 = np.zeros(simulation_size_y)
    gj2 = np.ones(simulation_size_y)
    gj3 = np.ones(simulation_size_y)
    fj1 = np.zeros(simulation_size_y)
    fj2 = np.ones(simulation_size_y)
    fj3 = np.ones(simulation_size_y)
    gk1 = np.zeros(simulation_size_z)
    gk2 = np.ones(simulation_size_z)
    gk3 = np.ones(simulation_size_z)
    fk1 = np.zeros(simulation_size_z)
    fk2 = np.ones(simulation_size_z)
    fk3 = np.ones(simulation_size_z)

    for n in range(npml):
        xxn = (npml - n) / npml
        xn = 0.33 * (xxn**3)
        fi1[n] = xn
        fi1[simulation_size_x - n - 1] = xn
        gi2[n] = 1 / (1 + xn)
        gi2[simulation_size_x - 1 - n] = 1 / (1 + xn)
        gi3[n] = (1 - xn) / (1 + xn)
        gi3[simulation_size_x - 1 - n] = (1 - xn) / (1 + xn)
        fj1[n] = xn
        fj1[simulation_size_y - n - 1] = xn
        gj2[n] = 1 / (1 + xn)
        gj2[simulation_size_y - 1 - n] = 1 / (1 + xn)
        gj3[n] = (1 - xn) / (1 + xn)
        gj3[simulation_size_y - 1 - n] = (1 - xn) / (1 + xn)
        fk1[n] = xn
        fk1[simulation_size_z - n - 1] = xn
        gk2[n] = 1 / (1 + xn)
        gk2[simulation_size_z - 1 - n] = 1 / (1 + xn)
        gk3[n] = (1 - xn) / (1 + xn)
        gk3[simulation_size_z - 1 - n] = (1 - xn) / (1 + xn)
        xxn = (npml - n - 0.5) / npml
        xn = 0.33 * (xxn**3)
        gi1[n] = xn
        gi1[simulation_size_x - 1 - n] = xn
        fi2[n] = 1 / (1 + xn)
        fi2[simulation_size_x - 1 - n] = 1 / (1 + xn)
        fi3[n] = (1 - xn) / (1 + xn)
        fi3[simulation_size_x - 1 - n] = (1 - xn) / (1 + xn)
        gj1[n] = xn
        gj1[simulation_size_y - 1 - n] = xn
        fj2[n] = 1 / (1 + xn)
        fj2[simulation_size_y - 1 - n] = 1 / (1 + xn)
        fj3[n] = (1 - xn) / (1 + xn)
        fj3[simulation_size_y - 1 - n] = (1 - xn) / (1 + xn)
        gk1[n] = xn
        gk1[simulation_size_z - 1 - n] = xn
        fk2[n] = 1 / (1 + xn)
        fk2[simulation_size_z - 1 - n] = 1 / (1 + xn)
        fk3[n] = (1 - xn) / (1 + xn)
        fk3[simulation_size_z - 1 - n] = (1 - xn) / (1 + xn)

    return (
        gi1,
        gi2,
        gi3,
        fi1,
        fi2,
        fi3,
        gj1,
        gj2,
        gj3,
        fj1,
        fj2,
        fj3,
        gk1,
        gk2,
        gk3,
        fk1,
        fk2,
        fk3,
    )


@numba.jit(nopython=True)
def calculate_dx_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dx,
    iDx,
    Hy,
    Hz,
    gj3,
    gk3,
    gj2,
    gk2,
    gi1,
):
    """Calculate the Dx Field"""
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                curl_h = Hz[i, j, k] - Hz[i, j - 1, k] - Hy[i, j, k] + Hy[i, j, k - 1]
                iDx[i, j, k] = iDx[i, j, k] + curl_h
                Dx[i, j, k] = gj3[j] * gk3[k] * Dx[i, j, k] + gj2[j] * gk2[k] * (
                    0.5 * curl_h + gi1[i] * iDx[i, j, k]
                )
    return Dx, iDx


@numba.jit(nopython=True)
def calculate_dy_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dy,
    iDy,
    Hx,
    Hz,
    gi3,
    gk3,
    gi2,
    gk2,
    gj1,
):
    """Calculate the Dy Field"""
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                curl_h = Hx[i, j, k] - Hx[i, j, k - 1] - Hz[i, j, k] + Hz[i - 1, j, k]
                iDy[i, j, k] = iDy[i, j, k] + curl_h
                Dy[i, j, k] = gi3[i] * gk3[k] * Dy[i, j, k] + gi2[i] * gk2[k] * (
                    0.5 * curl_h + gj1[j] * iDy[i, j, k]
                )
    return Dy, iDy


@numba.jit(nopython=True)
def calculate_dz_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dz,
    iDz,
    Hx,
    Hy,
    gi3,
    gj3,
    gi2,
    gj2,
    gk1,
):
    """Calculate the Dz Field"""
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                curl_h = Hy[i, j, k] - Hy[i - 1, j, k] - Hx[i, j, k] + Hx[i, j - 1, k]
                iDz[i, j, k] = iDz[i, j, k] + curl_h
                Dz[i, j, k] = gi3[i] * gj3[j] * Dz[i, j, k] + gi2[i] * gj2[j] * (
                    0.5 * curl_h + gk1[k] * iDz[i, j, k]
                )
    return Dz, iDz


@numba.jit(nopython=True)
def calculate_e_fields(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dx,
    Dy,
    Dz,
    eps_x,
    eps_y,
    eps_z,
    conductivity_x,
    conductivity_y,
    conductivity_z,
    Ex,
    Ey,
    Ez,
    Ix,
    Iy,
    Iz,
):
    """Calculate the E field from the D field"""
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y):
            for k in range(0, simulation_size_z):
                Ex[i, j, k] = eps_x[i, j, k] * (Dx[i, j, k] - Ix[i, j, k])
                Ix[i, j, k] = Ix[i, j, k] + conductivity_x[i, j, k] * Ex[i, j, k]
                Ey[i, j, k] = eps_y[i, j, k] * (Dy[i, j, k] - Iy[i, j, k])
                Iy[i, j, k] = Iy[i, j, k] + conductivity_y[i, j, k] * Ey[i, j, k]
                Ez[i, j, k] = eps_z[i, j, k] * (Dz[i, j, k] - Iz[i, j, k])
                Iz[i, j, k] = Iz[i, j, k] + conductivity_z[i, j, k] * Ez[i, j, k]
    return Ex, Ey, Ez, Ix, Iy, Iz


@numba.jit(nopython=True)
def calculate_hx_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Hx,
    iHx,
    Ey,
    Ez,
    fi1,
    fj2,
    fk2,
    fj3,
    fk3,
):
    """Calculate the Hx field"""
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y - 1):
            for k in range(0, simulation_size_z - 1):
                curl_e = Ey[i, j, k + 1] - Ey[i, j, k] - Ez[i, j + 1, k] + Ez[i, j, k]
                iHx[i, j, k] = iHx[i, j, k] + curl_e
                Hx[i, j, k] = fj3[j] * fk3[k] * Hx[i, j, k] + fj2[j] * fk2[k] * 0.5 * (
                    curl_e + fi1[i] * iHx[i, j, k]
                )
    return Hx, iHx


@numba.jit(nopython=True)
def calculate_hy_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Hy,
    iHy,
    Ex,
    Ez,
    fj1,
    fi2,
    fk2,
    fi3,
    fk3,
):
    """Calculate the Hy field"""
    for i in range(0, simulation_size_x - 1):
        for j in range(0, simulation_size_y):
            for k in range(0, simulation_size_z - 1):
                curl_e = Ez[i + 1, j, k] - Ez[i, j, k] - Ex[i, j, k + 1] + Ex[i, j, k]
                iHy[i, j, k] = iHy[i, j, k] + curl_e
                Hy[i, j, k] = fi3[i] * fk3[k] * Hy[i, j, k] + fi2[i] * fk2[k] * 0.5 * (
                    curl_e + fj1[j] * iHy[i, j, k]
                )
    return Hy, iHy


@numba.jit(nopython=True)
def calculate_hz_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Hz,
    iHz,
    Ex,
    Ey,
    fk1,
    fi2,
    fj2,
    fi3,
    fj3,
):
    """Calculate the Hz field"""
    for i in range(0, simulation_size_x - 1):
        for j in range(0, simulation_size_y - 1):
            for k in range(0, simulation_size_z):
                curl_e = Ex[i, j + 1, k] - Ex[i, j, k] - Ey[i + 1, j, k] + Ey[i, j, k]
                iHz[i, j, k] = iHz[i, j, k] + curl_e
                Hz[i, j, k] = fi3[i] * fj3[j] * Hz[i, j, k] + fi2[i] * fj2[j] * 0.5 * (
                    curl_e + fk1[k] * iHz[i, j, k]
                )
    return Hz, iHz


@numba.jit(nopython=True)
def compute_sar_rms(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Ex_sq_sum,
    Ey_sq_sum,
    Ez_sq_sum,
    sigma_x,
    sigma_y,
    sigma_z,
    rho,
    n_samples,
):
    """
    Compute Specific Absorption Rate (SAR) using RMS values
    SAR = σ|E_rms|² / (2ρ)
    """
    sar = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

    for i in range(simulation_size_x):
        for j in range(simulation_size_y):
            for k in range(simulation_size_z):
                # RMS magnitude squared of electric field
                e_rms_sq = (
                    Ex_sq_sum[i, j, k] + Ey_sq_sum[i, j, k] + Ez_sq_sum[i, j, k]
                ) / n_samples

                # Average conductivity (isotropic assumption)
                sigma_avg = (
                    sigma_x[i, j, k] + sigma_y[i, j, k] + sigma_z[i, j, k]
                ) / 3.0

                # SAR computation
                if rho[i, j, k] > 0:
                    sar[i, j, k] = (sigma_avg * e_rms_sq) / (2.0 * rho[i, j, k])
                else:
                    sar[i, j, k] = 0.0

    return sar


# Simulation parameters
simulation_size_x = 100
simulation_size_y = 100
simulation_size_z = 100

# Computational domain boundaries
npml = 8
ia = npml
ib = simulation_size_x - npml - 1
ja = npml
jb = simulation_size_y - npml - 1
ka = npml
kb = simulation_size_z - npml - 1

# Field arrays
Ex = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ey = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ez = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Dx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Dy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Dz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Hx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Hy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Hz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iDx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iDy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iDz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iHx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iHy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iHz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ix = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Iy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Iz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

# Material properties
eps_x = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
eps_y = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
eps_z = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
conductivity_x = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
conductivity_y = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
conductivity_z = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
sigma_x = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
sigma_y = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
sigma_z = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
rho = np.ones((simulation_size_x, simulation_size_y, simulation_size_z)) * 1000.0

# Step size and time step
dx = 0.005  # 5 mm cell size
dt = dx / (2.0 * 3e8)  # Time step (Courant condition with safety factor)
epsz = 8.854e-12

# Operating frequency
freq = 100e6  # 100 MHz
omega = 2.0 * pi * freq

# Tissue properties at 100 MHz
# Format: [eps_r, sigma (S/m), rho (kg/m³)]
tissue_props = {
    "muscle": [80.0, 0.5, 1000.0],
    "fat": [5.5, 0.04, 920.0],
    "tumor": [60.0, 0.8, 1050.0],
    "bone": [12.0, 0.02, 1900.0],
}

# Create patient model
print("Creating patient model...")
center_x = (ia + ib) // 2
center_y = (ja + jb) // 2
center_z = (ka + kb) // 2

for i in range(ia, ib + 1):
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            # Distance from center
            xdist = (i - center_x) * dx
            ydist = (j - center_y) * dx
            zdist = (k - center_z) * dx
            dist = sqrt(xdist**2 + ydist**2 + zdist**2)

            # Determine tissue type
            if k < ka + (kb - ka) // 4:
                # Bone layer at bottom
                eps_r, sigma_val, rho_val = tissue_props["bone"]
            elif dist > 0.15:  # 15 cm from center - fat
                eps_r, sigma_val, rho_val = tissue_props["fat"]
            elif dist < 0.05:  # 5 cm radius - tumor
                eps_r, sigma_val, rho_val = tissue_props["tumor"]
            else:
                # Muscle (default)
                eps_r, sigma_val, rho_val = tissue_props["muscle"]

            # Set material properties for each field component
            eps = eps_r
            cond = sigma_val

            # X-component
            eps_x[i, j, k] = 1.0 / (eps + (cond * dt / epsz))
            conductivity_x[i, j, k] = cond * dt / epsz
            sigma_x[i, j, k] = cond

            # Y-component
            eps_y[i, j, k] = 1.0 / (eps + (cond * dt / epsz))
            conductivity_y[i, j, k] = cond * dt / epsz
            sigma_y[i, j, k] = cond

            # Z-component
            eps_z[i, j, k] = 1.0 / (eps + (cond * dt / epsz))
            conductivity_z[i, j, k] = cond * dt / epsz
            sigma_z[i, j, k] = cond

            rho[i, j, k] = rho_val

# Calculate PML parameters
print("Calculating PML parameters...")
(
    gi1,
    gi2,
    gi3,
    fi1,
    fi2,
    fi3,
    gj1,
    gj2,
    gj3,
    fj1,
    fj2,
    fj3,
    gk1,
    gk2,
    gk3,
    fk1,
    fk2,
    fk3,
) = calculate_pml_parameters(
    npml, simulation_size_x, simulation_size_y, simulation_size_z
)

# Time stepping
periods = 10  # Run for 10 periods
time_steps = int(periods / (freq * dt))

print(f"\nSimulation Parameters:")
print(f"  Grid size: {simulation_size_x} x {simulation_size_y} x {simulation_size_z}")
print(f"  Cell size: {dx*1000:.1f} mm")
print(f"  Time step: {dt*1e12:.2f} ps")
print(f"  Frequency: {freq/1e6:.0f} MHz")
print(f"  Time steps: {time_steps}")
print(f"  PML thickness: {npml} cells")

# Sigma 60 applicator antenna positions
# 4 antennas arranged around patient
antennas = [
    {
        "i": ia - 2,
        "j": (ja + jb) // 2,
        "k": (ka + kb) // 2,
        "phase": 0.0,
        "amp": 1.0,
    },  # Left
    {
        "i": ib + 2,
        "j": (ja + jb) // 2,
        "k": (ka + kb) // 2,
        "phase": pi / 2,
        "amp": 1.0,
    },  # Right
    {
        "i": (ia + ib) // 2,
        "j": ja - 2,
        "k": (ka + kb) // 2,
        "phase": pi,
        "amp": 1.0,
    },  # Front
    {
        "i": (ia + ib) // 2,
        "j": jb + 2,
        "k": (ka + kb) // 2,
        "phase": 3 * pi / 2,
        "amp": 1.0,
    },  # Back
]

# Storage for SAR computation (RMS values)
Ex_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ey_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ez_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
n_sar_samples = 0

print(f"\nRunning FDTD simulation ({time_steps} time steps)...")
print("Progress: ", end="", flush=True)

# Main FDTD Loop
for time_step in range(1, time_steps + 1):
    # Calculate D fields
    Dx, iDx = calculate_dx_field(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        Dx,
        iDx,
        Hy,
        Hz,
        gj3,
        gk3,
        gj2,
        gk2,
        gi1,
    )
    Dy, iDy = calculate_dy_field(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        Dy,
        iDy,
        Hx,
        Hz,
        gi3,
        gk3,
        gi2,
        gk2,
        gj1,
    )
    Dz, iDz = calculate_dz_field(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        Dz,
        iDz,
        Hx,
        Hy,
        gi3,
        gj3,
        gi2,
        gj2,
        gk1,
    )

    # Add Sigma 60 antenna sources (soft sources)
    for ant in antennas:
        if (
            0 <= ant["i"] < simulation_size_x
            and 0 <= ant["j"] < simulation_size_y
            and 0 <= ant["k"] < simulation_size_z
        ):
            source = ant["amp"] * sin(omega * time_step * dt + ant["phase"])
            Dz[ant["i"], ant["j"], ant["k"]] += source

    # Calculate E fields from D fields
    Ex, Ey, Ez, Ix, Iy, Iz = calculate_e_fields(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        Dx,
        Dy,
        Dz,
        eps_x,
        eps_y,
        eps_z,
        conductivity_x,
        conductivity_y,
        conductivity_z,
        Ex,
        Ey,
        Ez,
        Ix,
        Iy,
        Iz,
    )

    # Calculate H fields
    Hx, iHx = calculate_hx_field(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        Hx,
        iHx,
        Ey,
        Ez,
        fi1,
        fj2,
        fk2,
        fj3,
        fk3,
    )
    Hy, iHy = calculate_hy_field(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        Hy,
        iHy,
        Ex,
        Ez,
        fj1,
        fi2,
        fk2,
        fi3,
        fk3,
    )
    Hz, iHz = calculate_hz_field(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        Hz,
        iHz,
        Ex,
        Ey,
        fk1,
        fi2,
        fj2,
        fi3,
        fj3,
    )

    # Accumulate field squares for RMS SAR computation (last few periods)
    if time_step > time_steps - int(periods * 0.5):
        Ex_sq_sum += Ex * Ex
        Ey_sq_sum += Ey * Ey
        Ez_sq_sum += Ez * Ez
        n_sar_samples += 1

    # Progress indicator
    if time_step % (time_steps // 10) == 0:
        print(".", end="", flush=True)

print(" Done!")

# Compute SAR using RMS values
print("\nComputing SAR distribution...")
sar = compute_sar_rms(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Ex_sq_sum,
    Ey_sq_sum,
    Ez_sq_sum,
    sigma_x,
    sigma_y,
    sigma_z,
    rho,
    n_sar_samples,
)

# Compute RMS E-field for visualization
Ex_rms = np.sqrt(Ex_sq_sum / n_sar_samples)
Ey_rms = np.sqrt(Ey_sq_sum / n_sar_samples)
Ez_rms = np.sqrt(Ez_sq_sum / n_sar_samples)
E_mag_rms = np.sqrt(Ex_rms**2 + Ey_rms**2 + Ez_rms**2)

# Visualization
print("Generating visualizations...")
center_x_idx = simulation_size_x // 2
center_y_idx = simulation_size_y // 2
center_z_idx = simulation_size_z // 2

fig = plt.figure(figsize=(15, 10))

# SAR distribution (XY plane, center Z)
ax1 = fig.add_subplot(2, 3, 1)
sar_slice_xy = sar[:, :, center_z_idx]
im1 = ax1.imshow(sar_slice_xy.T, origin="lower", cmap="hot", aspect="auto")
ax1.set_title("SAR Distribution (XY plane, center Z)")
ax1.set_xlabel("X (cells)")
ax1.set_ylabel("Y (cells)")
plt.colorbar(im1, ax=ax1, label="SAR (W/kg)")

# SAR distribution (XZ plane, center Y)
ax2 = fig.add_subplot(2, 3, 2)
sar_slice_xz = sar[:, center_y_idx, :]
im2 = ax2.imshow(sar_slice_xz.T, origin="lower", cmap="hot", aspect="auto")
ax2.set_title("SAR Distribution (XZ plane, center Y)")
ax2.set_xlabel("X (cells)")
ax2.set_ylabel("Z (cells)")
plt.colorbar(im2, ax=ax2, label="SAR (W/kg)")

# SAR distribution (YZ plane, center X)
ax3 = fig.add_subplot(2, 3, 3)
sar_slice_yz = sar[center_x_idx, :, :]
im3 = ax3.imshow(sar_slice_yz.T, origin="lower", cmap="hot", aspect="auto")
ax3.set_title("SAR Distribution (YZ plane, center X)")
ax3.set_xlabel("Y (cells)")
ax3.set_ylabel("Z (cells)")
plt.colorbar(im3, ax=ax3, label="SAR (W/kg)")

# E-field magnitude RMS (XY plane)
ax4 = fig.add_subplot(2, 3, 4)
e_mag_xy = E_mag_rms[:, :, center_z_idx]
im4 = ax4.imshow(e_mag_xy.T, origin="lower", cmap="viridis", aspect="auto")
ax4.set_title("E-field Magnitude RMS (XY plane, center Z)")
ax4.set_xlabel("X (cells)")
ax4.set_ylabel("Y (cells)")
plt.colorbar(im4, ax=ax4, label="|E| (V/m)")

# SAR statistics
ax5 = fig.add_subplot(2, 3, 5)
sar_flat = sar[ia : ib + 1, ja : jb + 1, ka : kb + 1].flatten()
sar_flat = sar_flat[sar_flat > 0]
ax5.hist(sar_flat, bins=50, edgecolor="black")
ax5.set_xlabel("SAR (W/kg)")
ax5.set_ylabel("Frequency")
ax5.set_title("SAR Distribution Histogram")
ax5.set_yscale("log")

# SAR profile along center line
ax6 = fig.add_subplot(2, 3, 6)
sar_profile = sar[center_x_idx, center_y_idx, ka : kb + 1]
z_coords = np.arange(ka, kb + 1) * dx * 100  # Convert to cm
ax6.plot(z_coords, sar_profile, "b-", linewidth=2)
ax6.set_xlabel("Z position (cm)")
ax6.set_ylabel("SAR (W/kg)")
ax6.set_title("SAR Profile (Center X, Center Y)")
ax6.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_file = "chapter6_hyperthermia_simulation.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Results saved to: {output_file}")

# Print statistics
print("\n" + "=" * 70)
print("Simulation Statistics:")
print("=" * 70)
print(f"Maximum SAR: {np.max(sar):.4f} W/kg")
print(
    f"Mean SAR (computational domain): {np.mean(sar[ia:ib+1, ja:jb+1, ka:kb+1]):.4f} W/kg"
)
print(f"Maximum E-field magnitude (RMS): {np.max(E_mag_rms):.2f} V/m")
print(f"SAR samples averaged: {n_sar_samples}")
print("=" * 70)

plt.show()
