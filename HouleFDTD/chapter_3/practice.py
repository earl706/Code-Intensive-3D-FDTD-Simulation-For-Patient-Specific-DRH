import numpy as np
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

# simulation size
simulation_size_x = 500
simulation_size_y = 500

# material position
medium_x = int(simulation_size_x / 2 - 1)
medium_y = int(simulation_size_y / 2 - 1)
ia = 7
ja = 7
ib = simulation_size_x - ia - 1
jb = simulation_size_y - ja - 1

# field values
Ez = np.zeros((simulation_size_x, simulation_size_y))
Dz = np.zeros((simulation_size_x, simulation_size_y))
Iz = np.zeros((simulation_size_x, simulation_size_y))
Hx = np.zeros((simulation_size_x, simulation_size_y))
Hy = np.zeros((simulation_size_x, simulation_size_y))
iHx = np.zeros((simulation_size_x, simulation_size_y))
iHy = np.zeros((simulation_size_x, simulation_size_y))

# electric and magnetic incident, step size, time step
Ez_inc = np.zeros(simulation_size_y)
Hz_inc = np.zeros(simulation_size_y)
dx = 0.01
dt = dx / 6e8

# fourier transform
number_of_frequencies = 3
freq = np.array((50e6, 300e6, 700e6))
real_pt = np.zeros((number_of_frequencies, simulation_size_x, simulation_size_y))
imag_pt = np.zeros((number_of_frequencies, simulation_size_x, simulation_size_y))
real_in = np.zeros(number_of_frequencies)
imag_in = np.zeros(number_of_frequencies)
phase = np.zeros((number_of_frequencies, simulation_size_y))
amp = np.zeros((number_of_frequencies, simulation_size_y))
gaz = np.ones((simulation_size_x, simulation_size_y))
gbz = np.ones((simulation_size_x, simulation_size_y))

# specify dielectric cylinder
