import torch
import torch.nn as nn
from discriminator import Discriminator
from generator import Generator
import dataset

dataset.download()


# # Funções de perda e otimização
# generator = Generator()
# discriminator = Discriminator()

# criterion = nn.BCELoss()  # Loss binária para o discriminador
# reconstruction_loss = nn.MSELoss()  # Perda de reconstrução (MSE ou L1)

# optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
# optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# num_epochs = 100
# batch_size = 64
# dataloader = 
# # Treinamento - Alternância entre treinar G e D
# for epoch in range(num_epochs):
#     for real_images in dataloader:
#         # Atualizando o discriminador
#         optimizer_d.zero_grad()
#         real_labels = torch.ones(batch_size, 1)
#         fake_labels = torch.zeros(batch_size, 1)
        
#         # Treinando com imagens reais
#         outputs = discriminator(real_images)
#         loss_real = criterion(outputs, real_labels)
        
#         # Gerando imagens falsas e treinando o discriminador
#         fake_images = generator(real_images)
#         outputs = discriminator(fake_images.detach())
#         loss_fake = criterion(outputs, fake_labels)
        
#         # Perda total do discriminador
#         loss_d = loss_real + loss_fake
#         loss_d.backward()
#         optimizer_d.step()
        
#         # Atualizando o gerador
#         optimizer_g.zero_grad()
#         fake_images = generator(real_images)
#         outputs = discriminator(fake_images)
        
#         # Perda adversarial e de reconstrução
#         loss_g_adv = criterion(outputs, real_labels)
#         loss_g_recon = reconstruction_loss(fake_images, real_images)
#         loss_g = loss_g_adv + 0.001 * loss_g_recon  # Soma ponderada das perdas
#         loss_g.backward()
#         optimizer_g.step()
