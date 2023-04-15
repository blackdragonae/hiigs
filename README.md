# hiigs

This code is designed to streamline the computation of various cosmological parameters by utilizing a Bayesian methodology. This approach incorporates prior knowledge and uncertainties when estimating these parameters, yielding more robust and accurate results.

In this instance, the Bayesian methodology is implemented through a nested Markov Chain Monte Carlo (MCMC) approach, specifically MultiNest. MCMC is a widely used technique in Bayesian statistics for generating samples from complex, high-dimensional probability distributions. The nested MCMC approach, such as MultiNest, enhances the efficiency of the sampling process by adapting the proposal distribution during sampling, which assists in more effectively exploring the parameter space.

The code relies on data from multiple cosmological tracers to calculate the cosmological parameters. Cosmological tracers are observable quantities that can provide information about the underlying cosmological model and its parameters. These tracers encompass cosmic microwave background radiation (CMB), Type Ia supernovae (SNIa), baryon acoustic oscillations (BAO), and HII galaxies (HIIG). By amalgamating data from various tracers, the code can more effectively constrain the cosmological parameters and generate more accurate estimates.

Current version:v2

Built by `Ricardo Chavez <https://www.irya.unam.mx/gente/r.chavez/>`
Licensed under the 2-clause BSD license (see ``LICENSE``).
