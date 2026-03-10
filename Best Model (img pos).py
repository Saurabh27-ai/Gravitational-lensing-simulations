import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root

# Lens parameters
sigma_v = 215   # km/s
c = 3e5         # km/s
b=1.2832321823186796       
e=0.2998912555902977  
r_c=0.06462168346190446

# Grid
N = 800
theta_max = 5*b

x = np.linspace(-theta_max, theta_max, N)
y = np.linspace(-theta_max, theta_max, N)

X, Y = np.meshgrid(x, y)

#Some expressions
sigma = sigma_v**2/(2*np.sqrt((1-e**2)*X**2 + Y**2 + r_c**2))   # Projected density, keeping G=1 units
k= b/(2*np.sqrt((1-e**2)*X**2 + Y**2 + r_c**2))

psi = np.sqrt((1-e**2)*X**2 + Y**2 + r_c**2)  # Potential term for SIE deflection, found by solving the Poisson equation

# SIE deflection, standard expressions taken from literature
alpha_x = (b*np.sqrt(1-e**2)/e) * np.arctan(e*X/(psi + r_c))
alpha_y = (b*np.sqrt(1-e**2)/e) * np.arctanh(e*Y/(psi + (1-e**2)*r_c))

# Jacobian matrix
dx = x[1] - x[0]
dy = y[1] - y[0]

dax_dy, dax_dx = np.gradient(alpha_x, dy, dx)
day_dy, day_dx = np.gradient(alpha_y, dy, dx)

detA = (1 - dax_dx)*(1 - day_dy) - dax_dy*day_dx

# Critical curves
plt.figure(figsize=(6,6))
plt.contour(X, Y, detA, levels=[0], colors='cyan')
plt.xlabel(r"$\theta_x$ (arcsec)")
plt.ylabel(r"$\theta_y$ (arcsec)")
plt.title("Critical Curves")
plt.axis('equal')
plt.grid(True)
plt.show()

# Caustic curves
beta_x = X - alpha_x
beta_y = Y - alpha_y

plt.figure(figsize=(6,6))
plt.contour(beta_x, beta_y, detA, levels=[0], colors='red')
plt.xlabel(r"$\beta_x$ (arcsec)")
plt.ylabel(r"$\beta_y$ (arcsec)")
plt.title("Caustic Curves")
plt.axis('equal')
plt.grid(True)
plt.show()

# Source position
beta_x0 = 0.02
beta_y0 = 0.0

# Interpolators
alpha_x_interp = RegularGridInterpolator((y, x), alpha_x, bounds_error=False, fill_value=None)
alpha_y_interp = RegularGridInterpolator((y, x), alpha_y, bounds_error=False, fill_value=None)

# Lens equation
def lens_equation(theta):

    tx, ty = theta

    ax = alpha_x_interp((ty, tx)).item()
    ay = alpha_y_interp((ty, tx)).item()

    fx = tx - ax - beta_x0
    fy = ty - ay - beta_y0

    return [fx, fy]


# Initial guesses


guesses = []

grid = np.linspace(-1.5*b, 1.5*b, 50)

for gx in grid:
    for gy in grid:
        guesses.append([gx, gy])

# -----------------------------
# Root finding
# -----------------------------

roots = []

for g in guesses:

    sol = root(lens_equation, g)

    if sol.success:

        r = sol.x

        # Remove duplicate solutions
        if not any(np.linalg.norm(r - np.array(rr)) < 1e-3 for rr in roots):
            roots.append(r)

roots = np.array(roots)

print("\nImage positions (arcsec):")
print(roots)

# -----------------------------
# Plot image positions
# -----------------------------

plt.figure(figsize=(6,6))

plt.contour(X, Y, detA, levels=[0], colors='cyan')

plt.scatter(roots[:,0], roots[:,1], color='red', s=80, label="Images")

plt.xlabel(r"$\theta_x$ (arcsec)")
plt.ylabel(r"$\theta_y$ (arcsec)")
plt.title("Image Positions from Lens Equation")

plt.axis('equal')
plt.grid(True)
plt.legend()

plt.show()


