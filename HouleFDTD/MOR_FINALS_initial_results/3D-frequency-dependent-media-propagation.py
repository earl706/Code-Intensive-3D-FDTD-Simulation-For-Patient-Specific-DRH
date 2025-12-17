"""
3D FDTD simulation of a plane wave on a dielectric sphere
"""

from math import exp, sqrt, cos, sin
import numba
import numpy as np
from matplotlib import pyplot as plt, animation


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


# simulation size
simulation_size_x = 180
simulation_size_y = 180
simulation_size_z = 180

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

# Specify the dielectric sphere
epsilon = np.ones(2)
sigma = np.zeros(2)
epsilon[1] = 30
sigma[1] = 0.7
radius = 30

for i in range(ia, ib + 1):
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            eps = epsilon[0]
            cond = sigma[0]
            xdist = source_x - i - 0.5
            ydist = source_y - j
            zdist = source_z - k
            dist = sqrt(xdist**2 + ydist**2 + zdist**2)
            if dist <= radius:
                eps = epsilon[1]
                cond = sigma[1]
            eps_x[i, j, k] = 1 / (eps + (cond * dt / epsz))
            conductivity_x[i, j, k] = cond * dt / epsz

for i in range(ia, ib + 1):
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            eps = epsilon[0]
            cond = sigma[0]
            xdist = source_x - i
            ydist = source_y - j - 0.5
            zdist = source_z - k
            dist = sqrt(xdist**2 + ydist**2 + zdist**2)
            if dist <= radius:
                eps = epsilon[1]
                cond = sigma[1]
            eps_y[i, j, k] = 1 / (eps + (cond * dt / epsz))
            conductivity_y[i, j, k] = cond * dt / epsz

for i in range(ia, ib + 1):
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            eps = epsilon[0]
            cond = sigma[0]
            xdist = source_x - i
            ydist = source_y - j
            zdist = source_z - k - 0.5
            dist = sqrt(xdist**2 + ydist**2 + zdist**2)
            if dist <= radius:
                eps = epsilon[1]
                cond = sigma[1]
            eps_z[i, j, k] = 1 / (eps + (cond * dt / epsz))
            conductivity_z[i, j, k] = cond * dt / epsz

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
time_steps = 1000

E_frames = []

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
    E_frames.append(Ez.copy())

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


# # Plot Fig. 4.7
# plt.rcParams["font.size"] = 12
# plt.rcParams["grid.color"] = "gray"
# plt.rcParams["grid.linestyle"] = "dotted"
# fig = plt.figure(figsize=(8, 7))
# X, Y = np.meshgrid(range(simulation_size_y), range(simulation_size_x))
# compare_array = np.arange(-8.5, 10.5, step=1)
# x_array = np.arange(-20, 20, step=1)
# # The data here was generated with the 3D Bessel function expansion program
# compare_amp = np.array(
#     [
#         [
#             0.074,
#             0.070,
#             0.064,
#             0.059,
#             0.054,
#             0.049,
#             0.044,
#             0.038,
#             0.033,
#             0.028,
#             0.022,
#             0.017,
#             0.012,
#             0.007,
#             0.005,
#             0.007,
#             0.012,
#             0.017,
#             0.022,
#         ],
#         [
#             0.302,
#             0.303,
#             0.301,
#             0.294,
#             0.281,
#             0.263,
#             0.238,
#             0.208,
#             0.173,
#             0.135,
#             0.095,
#             0.057,
#             0.036,
#             0.056,
#             0.091,
#             0.126,
#             0.156,
#             0.182,
#             0.202,
#         ],
#         [
#             0.329,
#             0.344,
#             0.353,
#             0.346,
#             0.336,
#             0.361,
#             0.436,
#             0.526,
#             0.587,
#             0.589,
#             0.524,
#             0.407,
#             0.285,
#             0.244,
#             0.300,
#             0.357,
#             0.360,
#             0.304,
#             0.208,
#         ],
#     ]
# )


# def plot_amp(ax, data, compare, freq, scale):
#     """Plot the Fourier transform amplitude at a specific frequency"""
#     ax.plot(x_array, data, color="k", linewidth=1)
#     ax.plot(compare_array, compare, "ko", mfc="none", linewidth=1)
#     plt.xlabel("cm")
#     plt.ylabel("Amplitude")
#     plt.xticks(np.arange(-5, 10, step=5))
#     plt.xlim(-9, 9)
#     plt.yticks(np.arange(0, 1, step=scale / 2))
#     plt.ylim(0, scale)
#     ax.text(20, 0.6, "{} MHz".format(int(freq / 1e6)), horizontalalignment="center")


# # Plot the results of the Fourier transform at each of the frequencies
# scale = np.array((0.1, 0.5, 0.7))
# for m in range(number_of_frequencies):
#     ax = fig.add_subplot(3, 1, m + 1)
#     plot_amp(ax, amp[m, :], compare_amp[m], freq[m], scale[m])
# plt.tight_layout()
# plt.show()


# animate
frames = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(
    E_frames[0][:, :, source_x],
    cmap="jet",
    vmin=np.min(E_frames),
    vmax=np.max(E_frames),
)
cbar = fig.colorbar(im)
cbar.set_label("Ez")
ax.set_xlabel("x")
ax.set_ylabel("y")

for frame in E_frames:
    im = ax.imshow(frame[:, :, source_x], cmap="jet")
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
plt.show()
# ani.save("3D_frequency_dependent_medium.mp4", writer="ffmpeg", fps=60)
