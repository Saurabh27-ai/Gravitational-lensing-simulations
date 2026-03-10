import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root
from astropy.cosmology import Planck18 as cosmo

# ORIGINAL MODEL PARAMETERS
sigma_v = 215
c = 3e5
e_true = 0.3
z_l = 0.04
z_s = 1.69

D_s = cosmo.angular_diameter_distance(z_s).value
D_l = cosmo.angular_diameter_distance(z_l).value
D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
D_ls_Ds = D_ls / D_s

b_rad = 4*np.pi*(sigma_v/c)**2 * D_ls_Ds
rad_to_arcsec = 206265
b_true = b_rad * rad_to_arcsec

rc_true = 0.05*b_true

# BEST FIT PARAMETERS
b_fit = 1.2832321823186796
e_fit = 0.2998912555902977
rc_fit = 0.06462168346190446

# SOURCE POSITION
beta_x0 = 0.02
beta_y0 = 0.0

# GRID
N = 800
theta_max = 5*b_true

x = np.linspace(-theta_max, theta_max, N)
y = np.linspace(-theta_max, theta_max, N)

X, Y = np.meshgrid(x, y)

# FUNCTION TO FIND IMAGE POSITIONS
def find_images(b, e, r_c):

    psi = np.sqrt((1-e**2)*X**2 + Y**2 + r_c**2)

    alpha_x = (b*np.sqrt(1-e**2)/e) * np.arctan(e*X/(psi + r_c))
    alpha_y = (b*np.sqrt(1-e**2)/e) * np.arctanh(e*Y/(psi + (1-e**2)*r_c))

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    dax_dy, dax_dx = np.gradient(alpha_x, dy, dx)
    day_dy, day_dx = np.gradient(alpha_y, dy, dx)

    detA = (1 - dax_dx)*(1 - day_dy) - dax_dy*day_dx

    # interpolators

    alpha_x_interp = RegularGridInterpolator((y, x), alpha_x,bounds_error=False,fill_value=None)

    alpha_y_interp = RegularGridInterpolator((y, x), alpha_y,bounds_error=False,fill_value=None)

    def lens_equation(theta):

        tx, ty = theta

        ax = alpha_x_interp((ty, tx)).item()
        ay = alpha_y_interp((ty, tx)).item()

        return [tx - ax - beta_x0,
                ty - ay - beta_y0]

    # initial guesses

    guesses = []
    grid = np.linspace(-1.5*b, 1.5*b, 50)

    for gx in grid:
        for gy in grid:
            guesses.append([gx, gy])

    roots = []

    for g in guesses:

        sol = root(lens_equation, g)

        if sol.success:

            r = sol.x

            if not any(np.linalg.norm(r - np.array(rr)) < 1e-3 for rr in roots):
                roots.append(r)

    roots = np.array(roots)

    return roots, detA

# FIND IMAGE POSITIONS
roots_true, detA_true = find_images(b_true, e_true, rc_true)

roots_fit, detA_fit = find_images(b_fit, e_fit, rc_fit)

print("\nOriginal model images")
print(roots_true)

print("\nBest fit model images")
print(roots_fit)

# SORT IMAGES (to compare correctly)
roots_true = roots_true[np.argsort(roots_true[:,0])]
roots_fit = roots_fit[np.argsort(roots_fit[:,0])]

# DISTANCE FROM ORIGIN
r_true = np.sqrt(roots_true[:,0]**2 + roots_true[:,1]**2)
r_fit = np.sqrt(roots_fit[:,0]**2 + roots_fit[:,1]**2)

print("\nDistance from origin (true)")
print(r_true)

print("\nDistance from origin (fit)")
print(r_fit)

#Absolute ERROR
abs_error = np.abs(r_true - r_fit)

print("\nAbsolute error (arcsec)")
print(abs_error)

# PERCENTAGE ERROR

percent_error = abs_error / r_true * 100

print("\nPercentage absolute error (%)")
print(percent_error)

# PLOT BOTH MODELS
plt.figure(figsize=(7,7))

plt.contour(X, Y, detA_true, levels=[0], colors='cyan')

plt.scatter(roots_true[:,0], roots_true[:,1], color='red', s=100, label="Original Model")

plt.scatter(roots_fit[:,0], roots_fit[:,1],color='blue', marker='x', s=120,label="Best Fit Model")

plt.xlabel(r"$\theta_x$ (arcsec)")
plt.ylabel(r"$\theta_y$ (arcsec)")

plt.title("Image Position Comparison")

plt.axis('equal')
plt.grid(True)

plt.legend()

plt.show()