import matplotlib.pyplot as plt


# Plot the Euler's formula from -5pi to 5pi with 5000 points

import numpy as np

x = np.linspace(-5*np.pi, 5*np.pi, 5000)

y = np.exp(1j*x)

plt.plot(x, y.real, label='Real part')

plt.plot(x, y.imag, label='Imaginary part')

plt.xlabel('x')

plt.ylabel('y')

plt.legend()

plt.show()

# Plot the Euler's formula from -5pi to 5pi with 5000 points in 3D space using the real and imaginary parts as the x and y

# coordinates and the angle as the z coordinate

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.plot(y.real, y.imag, x)

ax.set_xlabel('Real part')

ax.set_ylabel('Imaginary part')

ax.set_zlabel('Angle')

plt.show()



