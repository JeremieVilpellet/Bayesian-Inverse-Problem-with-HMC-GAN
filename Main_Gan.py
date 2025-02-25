# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:55:02 2024

@author: paul-
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model_file import Generator, Critic
from trainer_file import WGANTrainer
from data_file import get_dataloader

def compute_prior_stats(data_path, num_files):
    """
    Calculer la moyenne et la variance des données de simulation comme prior cible.
    """
    tensors = []
    for i in range(num_files):
        if i != 224:
            tensor = torch.load(f"{data_path}/darcy_data_{i}.pt")  # Charger les fichiers simulés
            tensor[0] = torch.clamp(tensor[0], min=0.0, max=1.6)  # v1
            tensor[1] = torch.clamp(tensor[1], min=0.0, max=1.6)  # v2
            tensor[2] = torch.clamp(tensor[2], min=0.0, max=1.0)  # p
            tensor[3] = torch.clamp(tensor[3], min=0.0, max=2.5)
            tensors.append(tensor)
    
    tensors = torch.stack(tensors)  # Shape: (num_files, 4, 50, 50)
    mean_true = torch.mean(tensors, dim=0)  # Moyenne par point sur la grille
    var_true = torch.var(tensors, dim=0)  # Variance par point sur la grille
    
    return mean_true, var_true

def compute_convergence(n_epochs, data_path, num_files, latent_dim, batch_size, generator_class, critic_class, dataloader_class):
    """
    Fonction pour calculer et tracer la convergence des erreurs de moyenne et variance.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculer le prior cible
    mean_true, var_true = compute_prior_stats(data_path, num_files)
    
    # Stocker les erreurs
    rmse_mean_list = []
    rmse_var_list = []
    
    for epoch in range(1, n_epochs + 1, 10):
        print(f"Training GAN for {epoch} epochs...")
        #############################################
        
        #############################################
        generator = generator_class(z_dim=latent_dim).to(device)
        critic = critic_class().to(device)
        
        # Créer le dataloader
        dataloader = dataloader_class(
            data_path=data_path,
            num_files=num_files,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        # Entraîner le GAN pour `epoch` epochs
        trainer = WGANTrainer(
            generator=generator,
            critic=critic,
            dataloader=dataloader,
            z_dim=latent_dim,
            num_epochs=epoch,  # Nombre d'epochs variable
            n_critic=1,
            n_gen=1,
            lambda_gp=5,
            learning_rate=1e-4,
            device=device,
            fixed_noise_size=16,
            print_interval=100
        )
        trainer.train()
        
        z_samples = torch.randn(1000, latent_dim, device=device)
        generated_fields = []
        with torch.no_grad():
            for z in z_samples:
                gen_state = generator(z.view(1, -1))[0]
                generated_fields.append(gen_state.cpu().numpy())
        
        generated_fields = np.array(generated_fields)  # Shape: (1000, 4, 50, 50)
        
        mean_g = np.mean(generated_fields, axis=0)
        var_g = np.var(generated_fields, axis=0)
        
        
        mean_error = np.sqrt(np.sum((mean_g - mean_true.cpu().numpy()) ** 2)) / np.sqrt(np.sum(mean_true.cpu().numpy() ** 2))
        var_error = np.sqrt(np.sum((var_g - var_true.cpu().numpy()) ** 2)) / np.sqrt(np.sum(var_true.cpu().numpy() ** 2))
        
        rmse_mean_list.append(mean_error)
        rmse_var_list.append(var_error)
        
    return rmse_mean_list, rmse_var_list





## Train the GAN (offline phase)

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path="./datas"

gen = Generator(z_dim=150).to(device)
critic = Critic().to(device)

files = [f for f in os.listdir(data_path)]
dataloader = get_dataloader(
    data_path=data_path, 
    num_files=len(files),
    batch_size=16, 
    shuffle=True, 
    num_workers=0, 
    drop_last=True
)

# Instanciation du trainer
trainer = WGANTrainer(
    generator=gen,
    critic=critic,
    dataloader=dataloader,
    z_dim=150,
    num_epochs=50,
    n_critic=2,
    n_gen=1,
    lambda_gp=5,
    learning_rate=1e-5,
    device=device,
    fixed_noise_size=32,
    print_interval=100
)

trainer.train()
    

### This section is not mandatory but aim to repdouce the convergence of the two first moments of the prior distribution
 
mean_true, var_true = compute_prior_stats(data_path, len(files))
rmse_mean_list, rmse_var_list = compute_convergence(50, data_path, len(files), 150, 10, Generator, Critic, get_dataloader)
 


plt.figure(figsize=(8, 6))  # Taille de la figure
plt.plot(range(1, 50 + 1, 10), rmse_mean_list, label='Mean', linewidth=2, color='tab:blue')  # Courbe moyenne
plt.plot(range(1, 50 + 1, 10), rmse_var_list, label='Std', linewidth=2, color='tab:orange')  # Courbe variance

plt.yscale('log')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Convergence of Mean and Variance', fontsize=14)

plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.legend(fontsize=12, loc='upper right')

plt.tight_layout()

plt.show()

