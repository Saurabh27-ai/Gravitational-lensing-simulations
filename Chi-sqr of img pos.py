import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

np.random.seed(42)

X_obs = np.array([(6.18551800e-01, -1.03297021e+00),(-1.15656962e+00, -4.84033092e-14),(6.18551800e-01, 1.03297021e+00),( 1.19885193e+00 , 5.33412613e-13),(-2.36183477e-03 , 3.29498712e-19)])

theta_x_obs = X_obs[:,0]
theta_y_obs = X_obs[:,1]

#Parameters guessed
b_true =1.1908590801860757
rc_true = 0.01 * b_true
e_true = 0.2

sigma_pos = 0.01


# SIE deflection

def deflection_SIE(theta_x, theta_y, b, rc, e):

    psi = np.sqrt((1-e**2)*theta_x**2 + theta_y**2 + rc**2)

    alpha_x = (b*np.sqrt(1-e**2)/e) * np.arctan(e*theta_x/(psi + rc))
    alpha_y = (b*np.sqrt(1-e**2)/e) * np.arctanh(e*theta_y/(psi + (1-e**2)*rc))

    return alpha_x, alpha_y



# Source position
def source_position(theta_x, theta_y, b, rc, e):

    ax, ay = deflection_SIE(theta_x, theta_y, b, rc, e)

    beta_x = theta_x - ax
    beta_y = theta_y - ay

    return beta_x, beta_y

# Magnification tensor

def magnification_tensor(theta_x, theta_y, b, rc, e):

    eps = 1e-5

    ax, ay = deflection_SIE(theta_x, theta_y, b, rc, e)

    ax_dx, ay_dx = deflection_SIE(theta_x + eps, theta_y, b, rc, e)
    ax_dy, ay_dy = deflection_SIE(theta_x, theta_y + eps, b, rc, e)

    dax_dx = (ax_dx - ax)/eps
    dax_dy = (ax_dy - ax)/eps

    day_dx = (ay_dx - ay)/eps
    day_dy = (ay_dy - ay)/eps

    A = np.array([[1 - dax_dx, -dax_dy],[-day_dx, 1 - day_dy]])

    mu = np.linalg.inv(A)

    return mu



# Chi-square 

def chi_square(params):

    b, rc, e = params

    beta_x, beta_y = source_position(theta_x_obs, theta_y_obs, b, rc, e)

    u_obs = np.vstack([beta_x, beta_y]).T

    A = np.zeros((2,2))
    b_vec = np.zeros(2)

    for i in range(len(theta_x_obs)):

      mu = magnification_tensor(theta_x_obs[i], theta_y_obs[i], b, rc, e)

      S_inv = np.eye(2) / sigma_pos**2

      W = mu.T @ S_inv @ mu

      A += W
      b_vec += W @ u_obs[i]

    u_mod = np.linalg.inv(A) @ b_vec
    
    chi2 = 0

    for i in range(len(theta_x_obs)):

        du = u_obs[i] - u_mod

        mu = magnification_tensor(theta_x_obs[i],theta_y_obs[i], b, rc, e)
        S_inv = np.eye(2)/sigma_pos**2

        chi2 += du.T @ S_inv @ du

    chi2 = chi2 / sigma_pos**2

    return chi2



# Likelihood

def log_likelihood(params):

    return -0.5 * chi_square(params)


# Priors

def log_prior(params):

    b, rc, e = params

    if 0.0 < b < 2.0 and 0.0 <= rc < 0.1*b and 0.0 < e < 1:
        return 0.0

    return -np.inf



# Posterior probability

def log_probability(params):

    lp = log_prior(params)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(params)



# MCMC 

ndim = 3
nwalkers = 32

initial = np.array([b_true, rc_true, e_true]) 

pos = initial + 1e-4*np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler( nwalkers,ndim,log_probability)

sampler.run_mcmc(pos, 5000, progress=True)

samples = sampler.get_chain(discard=1000,thin=10,flat=True)

# Corner plot

fig = corner.corner(samples,labels=["b", "r_c", "e"],truths=[b_true, rc_true, e_true], quantiles=[0.16, 0.5, 0.84], show_titles=True,title_fmt=".5f" )

plt.show()

# Best fit parameters

best = np.mean(samples, axis=0)

print("\nBest fit parameters:")
print("b =", best[0])
print("rc =", best[1])
print("e =", best[2])