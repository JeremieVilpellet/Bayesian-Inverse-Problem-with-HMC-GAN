import torch
import hamiltorch
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def latent_posterior(z, generator, obs_operator, observations,
                     prior_mean, prior_std, noise_mean, noise_std):
                     

    z_prior_score = torch.distributions.Normal(prior_mean,
                                               prior_std).log_prob(z).sum()

    gen_state = generator(z.view(1, len(z)))[0]
    gen_state = obs_operator(gen_state)
    
    error = observations - gen_state

    reconstruction_score = torch.distributions.Normal(noise_mean,
                                      noise_std).log_prob(error).sum()

    return z_prior_score + reconstruction_score

def hamiltonian_MC(z_init,posterior_params, HMC_params):
    posterior = lambda z: latent_posterior(z, **posterior_params)
    z_samples = hamiltorch.sample(log_prob_func=posterior,
                                  params_init=z_init,
                                  **HMC_params)
    return torch.stack(z_samples)



def compute_MAP(z, observations, generator, obs_operator, obs_std,
                num_iters=1000):
    
    z = z.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([z], lr=1e-2)

    #sloss = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=500,min_lr=0.001)
    with tqdm(range(num_iters), mininterval=3.,postfix=['Loss', dict(loss="0")]) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            gen_state = generator(z.view(1, -1))
            gen_state = gen_state[0]
            gen_obs = obs_operator(gen_state)
            error = (torch.linalg.norm(observations - gen_obs)**2) / (obs_std**2) + (torch.linalg.norm(z)**2)
            error.backward()
            optimizer.step()
            

            scheduler.step(error)
            pbar.postfix[1] = f"{error.item():.3f}"

    return z.detach()
