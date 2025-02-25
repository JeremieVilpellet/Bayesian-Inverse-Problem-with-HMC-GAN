# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:05:17 2024

@author: paul-
"""

import torch
import torch.optim as optim
from tqdm import tqdm

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1), device=device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp



  
class WGANTrainer:
    def __init__(
        self,
        generator,
        critic,
        dataloader,
        z_dim,
        num_epochs,
        n_critic=1,
        n_gen=1,
        lambda_gp=5,
        learning_rate=1e-4,
        device="cpu",
        fixed_noise_size=32,
        print_interval=100
    ):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.dataloader = dataloader
        self.z_dim = z_dim
        self.num_epochs = num_epochs
        self.n_critic = n_critic
        self.n_gen = n_gen
        self.lambda_gp = lambda_gp
        self.learning_rate = learning_rate
        self.device = device

        self.fixed_noise_size = fixed_noise_size
        self.print_interval = print_interval

        self.opt_gen = optim.RMSprop(self.generator.parameters(), lr=self.learning_rate)
        self.opt_critic = optim.RMSprop(self.critic.parameters(), lr=self.learning_rate)

        # Bruit fixe pour visualiser l'évolution du générateur
        self.fixed_noise = torch.randn(self.fixed_noise_size, self.z_dim, 1, 1, device=self.device)

        self.step = 0

        # Listes pour stocker les métriques et images
        self.images = []
        self.generator_losses = []
        self.critic_losses = []
        self.gradient_penalties = []

    def train(self):
        self.generator.train()
        self.critic.train()

        for epoch in range(self.num_epochs):
            for batch_idx, real in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.num_epochs}")):
                real = real.to(self.device)
                cur_batch_size = real.shape[0]

                # Mise à jour du critic n_critic fois
                for _ in range(self.n_critic):
                    noise = torch.randn(cur_batch_size, self.z_dim, 1, 1, device=self.device)
                    noise = noise.view(cur_batch_size, -1)
                    fake = self.generator(noise)
                    critic_real = self.critic(real).reshape(-1)
                    critic_fake = self.critic(fake).reshape(-1)
                    
                    gp = gradient_penalty(self.critic, real, fake, device=self.device)
                    
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.lambda_gp * gp

                    self.critic.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    self.opt_critic.step()

                # Mise à jour du générateur n_gen fois
                for _ in range(self.n_gen):
                    gen_noise = torch.randn(cur_batch_size, self.z_dim, 1, 1, device=self.device)
                    gen_noise = gen_noise.view(cur_batch_size, -1)
                    gen_fake = self.generator(gen_noise)
                    gen_out = self.critic(gen_fake).reshape(-1)
                    loss_gen = -torch.mean(gen_out)

                    self.generator.zero_grad()
                    loss_gen.backward()
                    self.opt_gen.step()

                # Enregistrement des métriques
                self.generator_losses.append(loss_gen.item())
                self.critic_losses.append(loss_critic.item())
                self.gradient_penalties.append(gp.item())
                
                torch.save(self.generator.state_dict(), "generator_opt_w.pt")
                torch.save(self.critic.state_dict(), "critic_opt_w.pt")

                # Affichage et génération d'images à intervalles réguliers
                if batch_idx % self.print_interval == 0 and batch_idx > 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_idx}/{len(self.dataloader)} "
                        f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}, GP: {gp:.4f}"
                    )

                    with torch.no_grad():
                        fixed_noise_flat = self.fixed_noise.view(self.fixed_noise_size, -1)
                        fake = self.generator(fixed_noise_flat)
                        fake_cpu = fake.detach().cpu()
                        self.images.append(fake_cpu)

                    self.step += 1

        print("Entraînement terminé.")
