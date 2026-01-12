import numpy as np
import matplotlib.pyplot as plt

theta_E= 2
beta_x=0.5
beta_y=0

N=2000
theta_max=10
x = np.linspace(-theta_max, theta_max, N)
y = np.linspace(-theta_max, theta_max, N)
X, Y = np.meshgrid(x, y)
 
eps = 1e-6
 
r2 = X**2 + Y**2 + eps
alpha_x = theta_E**2 * X/r2
alpha_y = theta_E**2 * Y/r2
 
beta_X = X - alpha_x
beta_Y = Y - alpha_y
 
source_radius = 0.088
source_mask = (beta_X - beta_x)**2 + (beta_Y - beta_y)**2 <= source_radius**2
 
detA = 1- (theta_E**4 /r2**2)
magnification = 1/np.abs(detA)

R= np.sqrt(X**2 + Y**2)

outer_mask= source_mask &(R> theta_E)
inner_mask = source_mask &(R < theta_E)

theta_outer_x = np.mean(X[outer_mask])
theta_outer_y = np.mean(Y[outer_mask])

theta_inner_x = np.mean(X[inner_mask])
theta_inner_y = np.mean(Y[inner_mask])

dtheta= (2*theta_max)/N
pixel_area = dtheta**2
source_area = np.pi*source_radius**2

mu_outer = np.sum(outer_mask)*pixel_area/source_area
mu_inner = np.sum(inner_mask)*pixel_area/source_area

print("Outer image position(theta+):", theta_outer_x, theta_outer_y)
print("Outer image magnification:", mu_outer)

print("Inner image position(theta-):", theta_inner_x, theta_inner_y)
print("Inner image magnification:", mu_inner)

print("Total magnification:", mu_outer + mu_inner)
 
plt.figure(figsize=(7,7))
plt.scatter(X[source_mask],Y[source_mask],c=magnification[source_mask], s=1, cmap= 'viridis')
 
phi= np.linspace(0, 2*np.pi, 1000)
plt.plot(theta_E*np.cos(phi), theta_E*np.sin(phi),'cyan', linestyle='--', linewidth=2, label='Einstein ring')
 
plt.scatter(0,0, color='blue',s=50, label= 'Lens')
 
plt.colorbar(label='Magnification')
plt.xlabel(r'$\theta_x$')
plt.ylabel(r'$\theta_y$')
plt.title('Point Mass Gravitational Lensing')
plt.axis('equal')
plt.legend()
plt.show()
 
 



