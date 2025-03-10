### Bayesian Inverse Problem with HMC-GAN in a physical context

""" All result are based on the article on the repo """

The goal of this project is to explore the theoretical foundations and practical applications of MCGAN (Multi-conditional Generative Adversarial Network) for Bayesian Inverse Problems in a physical context.

Data generation for the physical system (here, the Darcy flow problem in a unit square) is based on the finite element method, which is used to approximate the state of the system governed by a partial differential equation (PDE).

We use this data to train a WGAN (Wasserstein Generative Adversarial Network) with a conditional latent space generated by a Hamiltonian Monte Carlo method.
