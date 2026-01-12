import numpy as np
import matplotlib.pyplot as plt

theta_E = 2
beta_x = 2
beta_y = 0

N = 2000
theta_max = 5
x = np.linspace(-theta_max , theta_max , N)
y = np.linspace(-theta_max , theta_max , N)
X, Y = np.meshgrid(x, y)

eps = 1e-6
r = np.sqrt(X**2 + Y**2 + eps)
alpha_x = theta_E * X/r
alpha_y = theta_E * Y/r

beta_X = X - alpha_x
beta_Y = Y - alpha_y

source_radius =0.005
source_mask = (beta_X - beta_x)**2 + (beta_Y - beta_y)**2 <= source_radius**2

detA = 1 - theta_E/r
magnification = 1/np.abs(detA)

plt.figure(figsize=(7,7))
plt.scatter(X[source_mask], Y[source_mask], c= magnification[source_mask], s=1, cmap='viridis')

phi = np.linspace(0, 2*np.pi, 1000)
plt.plot(theta_E*np.cos(phi), theta_E*np.sin(phi),'cyan', linestyle='--', linewidth=2, label='Einstein radius')

plt.scatter(0,0, color='black', s=5, label= 'lens')

plt.colorbar(label='Magnification')
plt.xlabel(r'$\theta_x$')
plt.ylabel(r'$\theta_y$')
plt.title('Singular Isothermal Sphere Lensing : Inner and outer images')
plt.axis('equal')
plt.legend()
plt.show()