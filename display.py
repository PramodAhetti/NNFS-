import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the equationdef equation(x, y):
def equation(x,y):
    a = 22.7054 * x**2
    b = -44.6419 * x * y
    c = 68.0480 * y**2
    d = -32.0090 * x
    e = 71.6052
    output = np.maximum(0, a + b + c + d + e)
    return output

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Compute the output values
Z = equation(X, Y)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Add points at (0,0) and (1,1)
ax.scatter(0, 0, equation(0, 0), color='red', s=100, label='Point (0,0)')
ax.scatter(1, 1, equation(1, 1), color='blue', s=100, label='Point (1,1)')

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('output')
ax.set_title('Graph of the Equation')

# Change the viewing angle
ax.view_init(elev=30, azim=0)

# Add legend
ax.legend()
plt.savefig('output.png')
plt.show()
