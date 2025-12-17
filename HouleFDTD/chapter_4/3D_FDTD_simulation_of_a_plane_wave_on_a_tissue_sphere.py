"""
3D FDTD simulation of a plane wave on a tissue sphere
"""

from math import exp, sqrt, cos, sin
import numba
import numpy as np
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# functions for main FDTD loop
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
def calculate_inc_dy_field(ia, ib, ja, jb, ka, kb, Dy, hx_inc):
    """Calculate the incident Dy Field"""
    for i in range(ia, ib + 1):
        for j in range(ja, jb + 1):
            Dy[i, j, ka] = Dy[i, j, ka] - 0.5 * hx_inc[j]
            Dy[i, j, kb + 1] = Dy[i, j, kb + 1] + 0.5 * hx_inc[j]
    return Dy


@numba.jit(nopython=True)
def calculate_inc_dz_field(ia, ib, ja, jb, ka, kb, Dz, hx_inc):
    """Calculate the incident Dz Field"""
    for i in range(ia, ib + 1):
        for k in range(ka, kb + 1):
            Dz[i, ja, k] = Dz[i, ja, k] + 0.5 * hx_inc[ja - 1]
            Dz[i, jb, k] = Dz[i, jb, k] - 0.5 * hx_inc[jb]
    return Dz


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
def calculate_fourier_transform_ex(
    simulation_size_x,
    simulation_size_y,
    number_of_frequencies,
    real_pt,
    imag_pt,
    Ez,
    arg,
    time_step,
    source_z,
):
    """Calculate the Fourier transform of Ex"""
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y):
            for m in range(0, number_of_frequencies):
                real_pt[m, i, j] = (
                    real_pt[m, i, j] + cos(arg[m] * time_step) * Ez[i, j, source_z]
                )
                imag_pt[m, i, j] = (
                    imag_pt[m, i, j] - sin(arg[m] * time_step) * Ez[i, j, source_z]
                )
    return real_pt, imag_pt


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
def calculate_hx_inc(simulation_size_y, hx_inc, ez_inc):
    """Calculate incident Hx field"""
    for j in range(0, simulation_size_y - 1):
        hx_inc[j] = hx_inc[j] + 0.5 * (ez_inc[j] - ez_inc[j + 1])
    return hx_inc


@numba.jit(nopython=True)
def calculate_hx_with_incident_field(ia, ib, ja, jb, ka, kb, Hx, ez_inc):
    """Calculate Hx with incident Ez"""
    for i in range(ia, ib + 1):
        for k in range(ka, kb + 1):
            Hx[i, ja - 1, k] = Hx[i, ja - 1, k] + 0.5 * ez_inc[ja]
            Hx[i, jb, k] = Hx[i, jb, k] - 0.5 * ez_inc[jb]
    return Hx


@numba.jit(nopython=True)
def calculate_hy_with_incident_field(ia, ib, ja, jb, ka, kb, Hy, ez_inc):
    """Calculate Hy with incident Ez"""
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            Hy[ia - 1, j, k] = Hy[ia - 1, j, k] - 0.5 * ez_inc[j]
            Hy[ib, j, k] = Hy[ib, j, k] + 0.5 * ez_inc[j]
    return Hy


@numba.jit(nopython=True)
def accumulate_e_field_squared(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Ex,
    Ey,
    Ez,
    Ex_sq_sum,
    Ey_sq_sum,
    Ez_sq_sum,
):
    """Accumulate squared electric field components for SAR calculation"""
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y):
            for k in range(0, simulation_size_z):
                Ex_sq_sum[i, j, k] = Ex_sq_sum[i, j, k] + Ex[i, j, k] ** 2
                Ey_sq_sum[i, j, k] = Ey_sq_sum[i, j, k] + Ey[i, j, k] ** 2
                Ez_sq_sum[i, j, k] = Ez_sq_sum[i, j, k] + Ez[i, j, k] ** 2
    return Ex_sq_sum, Ey_sq_sum, Ez_sq_sum


@numba.jit(nopython=True)
def compute_instantaneous_sar(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Ex,
    Ey,
    Ez,
    sigma_x,
    sigma_y,
    sigma_z,
    rho,
):
    """
    Compute instantaneous SAR from current E-field values
    SAR = σ|E|² / (2ρ)
    Units: W/kg
    """
    sar = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

    for i in range(simulation_size_x):
        for j in range(simulation_size_y):
            for k in range(simulation_size_z):
                # Instantaneous magnitude squared of electric field
                e_mag_sq = Ex[i, j, k] ** 2 + Ey[i, j, k] ** 2 + Ez[i, j, k] ** 2

                # Average conductivity (isotropic assumption)
                sigma_avg = (
                    sigma_x[i, j, k] + sigma_y[i, j, k] + sigma_z[i, j, k]
                ) / 3.0

                # SAR computation
                if rho[i, j, k] > 0:
                    sar[i, j, k] = (sigma_avg * e_mag_sq) / (2.0 * rho[i, j, k])
                else:
                    sar[i, j, k] = 0.0

    return sar


@numba.jit(nopython=True)
def compute_sar(
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
    Units: W/kg
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


# simulation size
simulation_size_x = 200
simulation_size_y = 200
simulation_size_z = 200

# source position
source_x = int(simulation_size_x / 2)
source_y = int(simulation_size_y / 2)
source_z = int(simulation_size_z / 2)
ia = 7
ja = 7
ka = 7
ib = simulation_size_x - ia - 1
jb = simulation_size_y - ja - 1
kb = simulation_size_z - ka - 1

# field values
Ex = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ey = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ez = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ix = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Iy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Iz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Dx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Dy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Dz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iDx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iDy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iDz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Hx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Hy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Hz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iHx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iHy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
iHz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

# permittivity, conductivity, Hx/Ez incident
eps_x = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
eps_y = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
eps_z = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
conductivity_x = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
conductivity_y = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
conductivity_z = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
# For SAR calculation: actual conductivity (S/m) and density (kg/m³)
sigma_x = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
sigma_y = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
sigma_z = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
rho = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
hx_inc = np.zeros(simulation_size_y)
ez_inc = np.zeros(simulation_size_y)

# step size, time step
dx = 0.01  # Cell size
dt = dx / 6e8  # Time step size

# frequencies
epsz = 8.854e-12
number_of_frequencies = 3
freq = np.array((50e6, 200e6, 500e6))

# fourier transform
arg = 2 * np.pi * freq * dt
real_in = np.zeros(number_of_frequencies)
imag_in = np.zeros(number_of_frequencies)
real_pt = np.zeros(
    (number_of_frequencies, simulation_size_x, simulation_size_y, simulation_size_z)
)
imag_pt = np.zeros(
    (number_of_frequencies, simulation_size_x, simulation_size_y, simulation_size_z)
)
amp = np.zeros((number_of_frequencies, simulation_size_y))

# Specify the tissue sphere with multiple tissue types
# Tissue properties at ~100 MHz
# Format: [eps_r (relative permittivity), sigma (S/m), rho (kg/m³)]
tissue_props = {
    "free_space": [1.0, 0.0, 0.0],  # Free space (no density for SAR)
    "muscle": [80.0, 0.5, 1000.0],  # Muscle tissue
    "fat": [5.5, 0.04, 920.0],  # Fat tissue
    "tumor": [60.0, 0.8, 1050.0],  # Tumor tissue
    "bone": [12.0, 0.02, 1900.0],  # Bone tissue
}

# Tissue sphere geometry
outer_radius = 20  # Outer radius of tissue sphere (cells)
tumor_radius = 10  # Inner tumor radius (cells)

# Set up tissue properties for X-component
for i in range(ia, ib + 1):
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            # Distance from center (accounting for Yee grid offset for Ex)
            xdist = source_x - i - 0.5
            ydist = source_y - j
            zdist = source_z - k
            dist = sqrt(xdist**2 + ydist**2 + zdist**2)
            
            # Determine tissue type based on distance
            if dist <= tumor_radius:
                # Tumor at center
                eps_r, sigma_val, rho_val = tissue_props["tumor"]
            elif dist <= outer_radius:
                # Muscle tissue surrounding tumor
                eps_r, sigma_val, rho_val = tissue_props["muscle"]
            else:
                # Free space
                eps_r, sigma_val, rho_val = tissue_props["free_space"]
            
            eps_x[i, j, k] = 1 / (eps_r + (sigma_val * dt / epsz))
            conductivity_x[i, j, k] = sigma_val * dt / epsz
            sigma_x[i, j, k] = sigma_val
            rho[i, j, k] = rho_val

# Set up tissue properties for Y-component
for i in range(ia, ib + 1):
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            # Distance from center (accounting for Yee grid offset for Ey)
            xdist = source_x - i
            ydist = source_y - j - 0.5
            zdist = source_z - k
            dist = sqrt(xdist**2 + ydist**2 + zdist**2)
            
            # Determine tissue type based on distance
            if dist <= tumor_radius:
                # Tumor at center
                eps_r, sigma_val, rho_val = tissue_props["tumor"]
            elif dist <= outer_radius:
                # Muscle tissue surrounding tumor
                eps_r, sigma_val, rho_val = tissue_props["muscle"]
            else:
                # Free space
                eps_r, sigma_val, rho_val = tissue_props["free_space"]
            
            eps_y[i, j, k] = 1 / (eps_r + (sigma_val * dt / epsz))
            conductivity_y[i, j, k] = sigma_val * dt / epsz
            sigma_y[i, j, k] = sigma_val

# Set up tissue properties for Z-component
for i in range(ia, ib + 1):
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            # Distance from center (accounting for Yee grid offset for Ez)
            xdist = source_x - i
            ydist = source_y - j
            zdist = source_z - k - 0.5
            dist = sqrt(xdist**2 + ydist**2 + zdist**2)
            
            # Determine tissue type based on distance
            if dist <= tumor_radius:
                # Tumor at center
                eps_r, sigma_val, rho_val = tissue_props["tumor"]
            elif dist <= outer_radius:
                # Muscle tissue surrounding tumor
                eps_r, sigma_val, rho_val = tissue_props["muscle"]
            else:
                # Free space
                eps_r, sigma_val, rho_val = tissue_props["free_space"]
            
            eps_z[i, j, k] = 1 / (eps_r + (sigma_val * dt / epsz))
            conductivity_z[i, j, k] = sigma_val * dt / epsz
            sigma_z[i, j, k] = sigma_val

# Pulse Parameters
pulse_width = 8
pulse_delay = 20

# Calculate the PML parameters
npml = 8
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
boundary_low = [0, 0]
boundary_high = [0, 0]
time_steps = 500

# Arrays for SAR calculation (accumulate E-field squared)
Ex_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ey_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ez_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
sar_start_step = int(time_steps * 0.7)  # Start accumulating after 70% of simulation
n_sar_samples = 0  # Count samples for averaging

E_frames = []
SAR_frames = []  # Store instantaneous SAR frames for animation

# Main FDTD Loop
for time_step in range(1, time_steps + 1):
    # Calculate the incident buffer
    for j in range(1, simulation_size_y - 1):
        ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j - 1] - hx_inc[j])

    # Fourier transform of the incident field
    for m in range(number_of_frequencies):
        real_in[m] = real_in[m] + cos(arg[m] * time_step) * ez_inc[ja - 1]
        imag_in[m] = imag_in[m] - sin(arg[m] * time_step) * ez_inc[ja - 1]

    # Absorbing Boundary Conditions
    ez_inc[0] = boundary_low.pop(0)
    boundary_low.append(ez_inc[1])
    ez_inc[simulation_size_y - 1] = boundary_high.pop(0)
    boundary_high.append(ez_inc[simulation_size_y - 2])

    # Calculate the D Fields
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

    # Add the source at the gap
    pulse = exp(-0.5 * ((pulse_delay - time_step) / pulse_width) ** 2)
    ez_inc[3] = pulse
    Dy = calculate_inc_dy_field(ia, ib, ja, jb, ka, kb, Dy, hx_inc)
    Dz = calculate_inc_dz_field(ia, ib, ja, jb, ka, kb, Dz, hx_inc)

    # Calculate the E field from the D field
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

    # Accumulate E-field squared for SAR calculation (after field stabilizes)
    if time_step >= sar_start_step:
        Ex_sq_sum, Ey_sq_sum, Ez_sq_sum = accumulate_e_field_squared(
            simulation_size_x,
            simulation_size_y,
            simulation_size_z,
            Ex,
            Ey,
            Ez,
            Ex_sq_sum,
            Ey_sq_sum,
            Ez_sq_sum,
        )
        n_sar_samples += 1

    # Calculate the Fourier transform of Ex
    real_pt, imag_pt = calculate_fourier_transform_ex(
        simulation_size_x,
        simulation_size_y,
        number_of_frequencies,
        real_pt,
        imag_pt,
        Ez,
        arg,
        time_step,
        source_z,
    )

    # Calculate the H fields
    hx_inc = calculate_hx_inc(simulation_size_y, hx_inc, ez_inc)
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
    Hx = calculate_hx_with_incident_field(ia, ib, ja, jb, ka, kb, Hx, ez_inc)
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
    Hy = calculate_hy_with_incident_field(ia, ib, ja, jb, ka, kb, Hy, ez_inc)
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
    if time_step % 5 == 0:
        print(time_step)
        print(np.min(Ez), np.max(Ez))
    
    # Store E-field frame
    E_frames.append(Ez.copy())
    
    # Compute and store instantaneous SAR frame
    sar_instant = compute_instantaneous_sar(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        Ex,
        Ey,
        Ez,
        sigma_x,
        sigma_y,
        sigma_z,
        rho,
    )
    SAR_frames.append(sar_instant.copy())

# Calculate the Fourier amplitude of the incident pulse
amp_in = np.sqrt(real_in**2 + imag_in**2)

# Calculate the Fourier amplitude of the total field
for m in range(number_of_frequencies):
    for j in range(ja, jb + 1):
        if eps_z[source_x, j, source_z] < 1:
            amp[m, j] = (
                1
                / (amp_in[m])
                * sqrt(
                    real_pt[m, source_x, j, source_z] ** 2
                    + imag_pt[m, source_x, j, source_z] ** 2
                )
            )

# Compute SAR (Specific Absorption Rate)
print("\nComputing SAR distribution...")
print(f"SAR samples collected: {n_sar_samples}")
if n_sar_samples > 0:
    SAR = compute_sar(
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
    print(f"SAR statistics:")
    print(f"  Max SAR: {np.max(SAR):.4f} W/kg")
    if np.any(rho > 0):
        print(f"  Mean SAR (tissue only): {np.mean(SAR[rho > 0]):.4f} W/kg")
        # Tumor density is 1050.0 kg/m³
        tumor_mask = np.abs(rho - 1050.0) < 0.1
        if np.any(tumor_mask):
            print(f"  Max SAR in tumor region: {np.max(SAR[tumor_mask]):.4f} W/kg")
        # Muscle density is 1000.0 kg/m³
        muscle_mask = np.abs(rho - 1000.0) < 0.1
        if np.any(muscle_mask):
            print(f"  Max SAR in muscle region: {np.max(SAR[muscle_mask]):.4f} W/kg")
else:
    print("Warning: No SAR samples collected. SAR calculation skipped.")
    SAR = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))




# Function to prepare 3D voxel data for a frame
def prepare_3d_voxel_data(data, step=3, threshold_ratio=0.3):
    """Prepare 3D voxel data for visualization"""
    # Normalize data
    data_norm = np.abs(data)
    data_max = np.max(data_norm)
    if data_max == 0:
        return None, None, None, None, None
    
    threshold = data_max * threshold_ratio
    
    # Downsample data
    data_sub = data_norm[::step, ::step, ::step]
    nx, ny, nz = data_sub.shape
    
    # Create coordinate arrays - voxels expects coordinates to be one element larger
    x_edges = np.arange(nx + 1) * step
    y_edges = np.arange(ny + 1) * step
    z_edges = np.arange(nz + 1) * step
    X_sub, Y_sub, Z_sub = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')
    
    # Create voxel plot for values above threshold
    voxels = data_sub > threshold
    
    # Create color array based on data values
    colors = np.empty(voxels.shape + (4,))
    data_normalized = np.clip(data_sub / data_max, 0, 1)
    
    return X_sub, Y_sub, Z_sub, voxels, colors, data_normalized, data_max

# Function to create 3D isometric view (static)
def create_3d_isometric_view(data, title, cmap="jet", threshold_ratio=0.3):
    """Create a 3D isometric view using isosurfaces"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    result = prepare_3d_voxel_data(data, step=3, threshold_ratio=threshold_ratio)
    if result[0] is None:
        print(f"Warning: {title} has zero data, skipping 3D view")
        return fig, ax
    
    X_sub, Y_sub, Z_sub, voxels, colors, data_normalized, data_max = result
    
    if not np.any(voxels):
        print(f"Warning: No voxels above threshold for {title}")
        return fig, ax
    
    # Use colormap-like coloring (red to yellow for hot, blue to red for jet)
    if cmap == "hot":
        colors[..., 0] = 1.0  # Red
        colors[..., 1] = data_normalized  # Green (yellow when max)
        colors[..., 2] = 0.0  # Blue
    else:  # jet-like
        colors[..., 0] = np.clip(1.0 - 2 * data_normalized, 0, 1)  # Red
        colors[..., 1] = np.clip(2 * data_normalized, 0, 1)  # Green
        colors[..., 2] = np.clip(2 * (data_normalized - 0.5), 0, 1)  # Blue
    # Alpha based on magnitude (only for voxels that are True)
    colors[..., 3] = np.where(voxels, np.clip(data_normalized, 0.2, 0.8), 0.0)
    
    ax.voxels(X_sub, Y_sub, Z_sub, voxels, facecolors=colors, edgecolor='none', alpha=0.6)
    
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    ax.set_zlabel("Z (cells)")
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([data.shape[0], data.shape[1], data.shape[2]]).max() / 2.0
    mid_x = data.shape[0] / 2.0
    mid_y = data.shape[1] / 2.0
    mid_z = data.shape[2] / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return fig, ax

# Animated 3D isometric views side-by-side using ArtistAnimation
print("\nCreating animated 3D isometric views (E-field and SAR)...")
if len(E_frames) > 0 and len(SAR_frames) > 0:
    # Re-implement using ArtistAnimation (similar to reference file)
    # Create 2D projections for each frame (works better with ArtistAnimation)
    fig_anim = plt.figure(figsize=(20, 10))
    ax_e = fig_anim.add_subplot(1, 2, 1)
    ax_sar = fig_anim.add_subplot(1, 2, 2)
    
    # Find global min/max for consistent color scaling
    e_max = max(np.max(np.abs(frame)) for frame in E_frames)
    sar_max = max(np.max(frame) for frame in SAR_frames) if len(SAR_frames) > 0 else 1.0
    
    # Prepare frames for ArtistAnimation
    frames = []
    
    for frame_idx in range(len(E_frames)):
        # Get current frame data
        e_data = np.abs(E_frames[frame_idx])
        sar_data = SAR_frames[frame_idx]
        
        # Create maximum intensity projection for E-field (isometric-like view)
        # Project along Z-axis (max along depth)
        e_projection = np.max(e_data, axis=2)
        
        # Create maximum intensity projection for SAR
        sar_projection = np.max(sar_data, axis=2)
        
        # Plot E-field projection
        im_e = ax_e.imshow(
            e_projection,
            cmap="jet",
            origin="lower",
            vmin=0,
            vmax=e_max,
            animated=True
        )
        ax_e.set_title(f"E-field (Ez) - Frame {frame_idx + 1}/{len(E_frames)}")
        ax_e.set_xlabel("Y (cells)")
        ax_e.set_ylabel("X (cells)")
        
        # Plot SAR projection
        im_sar = ax_sar.imshow(
            sar_projection,
            cmap="hot",
            origin="lower",
            vmin=0,
            vmax=sar_max,
            animated=True
        )
        ax_sar.set_title(f"SAR Distribution - Frame {frame_idx + 1}/{len(SAR_frames)}")
        ax_sar.set_xlabel("Y (cells)")
        ax_sar.set_ylabel("X (cells)")
        
        frames.append([im_e, im_sar])
    
    # Create animation using ArtistAnimation (similar to reference file)
    ani = animation.ArtistAnimation(
        fig_anim, frames, interval=20, blit=True, repeat_delay=1000
    )
    
    plt.tight_layout()
    
    # Save animation as video (similar to reference file)
    print("\nSaving 2D animation as video...")
    ani.save("3D_FDTD_simulation_of_a_plane_wave_on_a_tissue_sphere_sar_2d.mp4", writer="ffmpeg", fps=60)
    print("✓ 2D animation saved")
    
    # Now create 3D isometric view animations using FuncAnimation
    print("\nCreating 3D isometric view animations...")
    
    # 3D E-field animation
    fig_3d_e = plt.figure(figsize=(12, 10))
    ax_3d_e = fig_3d_e.add_subplot(111, projection='3d')
    
    # Prepare coordinate arrays for 3D surface plots
    sample_data = E_frames[0]
    step_3d = 2  # Downsample for performance
    nx, ny = sample_data.shape[0], sample_data.shape[1]
    x_coords = np.arange(0, nx, step_3d)
    y_coords = np.arange(0, ny, step_3d)
    X_3d, Y_3d = np.meshgrid(y_coords, x_coords)
    
    # Set fixed axis limits
    x_min, x_max = 0, ny
    y_min, y_max = 0, nx
    z_min, z_max = 0, e_max
    
    def update_3d_e(frame_num):
        ax_3d_e.clear()
        e_data = np.abs(E_frames[frame_num])
        e_projection = np.max(e_data, axis=2)
        e_projection_3d = e_projection[::step_3d, ::step_3d]
        
        surf_e = ax_3d_e.plot_surface(
            X_3d, Y_3d, e_projection_3d,
            cmap="jet",
            vmin=0,
            vmax=e_max,
            alpha=0.9,
            linewidth=0,
            antialiased=True
        )
        ax_3d_e.set_xlabel("Y (cells)")
        ax_3d_e.set_ylabel("X (cells)")
        ax_3d_e.set_zlabel("Magnitude")
        ax_3d_e.set_title(f"E-field (Ez) 3D Isometric - Frame {frame_num + 1}/{len(E_frames)}")
        ax_3d_e.view_init(elev=30, azim=45)  # Isometric view angle
        # Set fixed axis limits
        ax_3d_e.set_xlim(x_min, x_max)
        ax_3d_e.set_ylim(y_min, y_max)
        ax_3d_e.set_zlim(z_min, z_max)
        
    ani_3d_e = animation.FuncAnimation(
        fig_3d_e, update_3d_e, frames=len(E_frames),
        interval=20, blit=False, repeat=True, repeat_delay=1000
    )
    
    print("Saving 3D E-field animation as video...")
    ani_3d_e.save("3D_FDTD_simulation_of_a_plane_wave_on_a_tissue_sphere_efield_3d.mp4", writer="ffmpeg", fps=60)
    print("✓ 3D E-field animation saved")
    
    # 3D SAR animation
    fig_3d_sar = plt.figure(figsize=(12, 10))
    ax_3d_sar = fig_3d_sar.add_subplot(111, projection='3d')
    
    # Set fixed axis limits for SAR
    sar_z_min, sar_z_max = 0, sar_max
    
    def update_3d_sar(frame_num):
        ax_3d_sar.clear()
        sar_data = SAR_frames[frame_num]
        sar_projection = np.max(sar_data, axis=2)
        sar_projection_3d = sar_projection[::step_3d, ::step_3d]
        
        surf_sar = ax_3d_sar.plot_surface(
            X_3d, Y_3d, sar_projection_3d,
            cmap="hot",
            vmin=0,
            vmax=sar_max,
            alpha=0.9,
            linewidth=0,
            antialiased=True
        )
        ax_3d_sar.set_xlabel("Y (cells)")
        ax_3d_sar.set_ylabel("X (cells)")
        ax_3d_sar.set_zlabel("SAR (W/kg)")
        ax_3d_sar.set_title(f"SAR Distribution 3D Isometric - Frame {frame_num + 1}/{len(SAR_frames)}")
        ax_3d_sar.view_init(elev=30, azim=45)  # Isometric view angle
        # Set fixed axis limits
        ax_3d_sar.set_xlim(x_min, x_max)
        ax_3d_sar.set_ylim(y_min, y_max)
        ax_3d_sar.set_zlim(sar_z_min, sar_z_max)
        
    ani_3d_sar = animation.FuncAnimation(
        fig_3d_sar, update_3d_sar, frames=len(SAR_frames),
        interval=20, blit=False, repeat=True, repeat_delay=1000
    )
    
    print("Saving 3D SAR animation as video...")
    ani_3d_sar.save("3D_FDTD_simulation_of_a_plane_wave_on_a_tissue_sphere_sar_3d.mp4", writer="ffmpeg", fps=60)
    print("✓ 3D SAR animation saved")
    
    plt.show()
elif len(E_frames) > 0:
    # Only E-field available
    print("Warning: SAR frames not available, showing only E-field animation")
    # Re-implement using ArtistAnimation (similar to reference file)
    fig_anim = plt.figure(figsize=(12, 10))
    ax_e = fig_anim.add_subplot(1, 1, 1)
    
    # Find global min/max for consistent color scaling
    e_max = max(np.max(np.abs(frame)) for frame in E_frames)
    
    # Prepare frames for ArtistAnimation
    frames = []
    
    for frame_idx in range(len(E_frames)):
        e_data = np.abs(E_frames[frame_idx])
        
        # Create maximum intensity projection (isometric-like view)
        e_projection = np.max(e_data, axis=2)
        
        im_e = ax_e.imshow(
            e_projection,
            cmap="jet",
            origin="lower",
            vmin=0,
            vmax=e_max,
            animated=True
        )
        ax_e.set_title(f"E-field (Ez) - Frame {frame_idx + 1}/{len(E_frames)}")
        ax_e.set_xlabel("Y (cells)")
        ax_e.set_ylabel("X (cells)")
        
        frames.append([im_e])
    
    # Create animation using ArtistAnimation (similar to reference file)
    ani = animation.ArtistAnimation(
        fig_anim, frames, interval=20, blit=True, repeat_delay=1000
    )
    
    # Save animation as video (similar to reference file)
    print("\nSaving 2D animation as video...")
    ani.save("3D_FDTD_simulation_of_a_plane_wave_on_a_tissue_sphere_2d.mp4", writer="ffmpeg", fps=60)
    print("✓ 2D animation saved")
    
    # Now create 3D isometric view animation using FuncAnimation
    print("\nCreating 3D isometric view animation...")
    
    fig_3d_e = plt.figure(figsize=(12, 10))
    ax_3d_e = fig_3d_e.add_subplot(111, projection='3d')
    
    # Prepare coordinate arrays for 3D surface plots
    sample_data = E_frames[0]
    step_3d = 2  # Downsample for performance
    nx, ny = sample_data.shape[0], sample_data.shape[1]
    x_coords = np.arange(0, nx, step_3d)
    y_coords = np.arange(0, ny, step_3d)
    X_3d, Y_3d = np.meshgrid(y_coords, x_coords)
    
    # Set fixed axis limits
    x_min, x_max = 0, ny
    y_min, y_max = 0, nx
    z_min, z_max = 0, e_max
    
    def update_3d_e(frame_num):
        ax_3d_e.clear()
        e_data = np.abs(E_frames[frame_num])
        e_projection = np.max(e_data, axis=2)
        e_projection_3d = e_projection[::step_3d, ::step_3d]
        
        surf_e = ax_3d_e.plot_surface(
            X_3d, Y_3d, e_projection_3d,
            cmap="jet",
            vmin=0,
            vmax=e_max,
            alpha=0.9,
            linewidth=0,
            antialiased=True
        )
        ax_3d_e.set_xlabel("Y (cells)")
        ax_3d_e.set_ylabel("X (cells)")
        ax_3d_e.set_zlabel("Magnitude")
        ax_3d_e.set_title(f"E-field (Ez) 3D Isometric - Frame {frame_num + 1}/{len(E_frames)}")
        ax_3d_e.view_init(elev=30, azim=45)  # Isometric view angle
        # Set fixed axis limits
        ax_3d_e.set_xlim(x_min, x_max)
        ax_3d_e.set_ylim(y_min, y_max)
        ax_3d_e.set_zlim(z_min, z_max)
        
    ani_3d_e = animation.FuncAnimation(
        fig_3d_e, update_3d_e, frames=len(E_frames),
        interval=20, blit=False, repeat=True, repeat_delay=1000
    )
    
    print("Saving 3D E-field animation as video...")
    ani_3d_e.save("3D_FDTD_simulation_of_a_plane_wave_on_a_tissue_sphere_efield_3d.mp4", writer="ffmpeg", fps=60)
    print("✓ 3D E-field animation saved")
    
    plt.show()

