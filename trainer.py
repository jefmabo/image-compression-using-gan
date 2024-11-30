import torch
import torch.nn as nn
import dataset as local_dt
import numpy as np
import state_handlers
import metrics
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset
from discriminator import Discriminator
from generator import Generator
from torchvision import transforms, datasets


def train():
    batch_size = 128
    num_epochs = 101

    file_paths = local_dt.get_all_files_path()

    if len(file_paths) <= 0:
        file_paths = local_dt.download()

    # Funções de perda e otimização
    generator = Generator()
    discriminator = Discriminator()

    criterion = nn.BCELoss()  # Loss binária para o discriminador
    reconstruction_loss = nn.MSELoss()  # Perda de reconstrução (MSE ou L1)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    transform = transforms.Compose([
        transforms.Resize((160, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(root='dataset', transform=transform)
    
    dataset_size_stages = [2048, 4096, 6144, 8192, 10240]

    for stage, num_samples in enumerate(dataset_size_stages):
        print(f"Starting stage {stage + 1}/{len(dataset_size_stages)} with {num_samples} samples...")

        index = list(range(num_samples))
        subset = Subset(dataset, index)
        dataloader = DataLoader(subset, batch_size, shuffle=True, num_workers=0)

        # Treinamento - Alternância entre treinar G e D
        for epoch in range(num_epochs // len(dataset_size_stages)):
            num_samples_processed = 0

            for real_images, _ in dataloader:

                num_samples_processed += len(real_images)

                # Atualizando o discriminador
                optimizer_d.zero_grad()
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                # Treinando com imagens reais
                outputs = discriminator(real_images)
                
                # Aplica uma media para reduzir as dimensoes do output
                outputs = torch.mean(outputs, dim=[2, 3])
                loss_real = criterion(outputs, real_labels)
                
                # Gerando imagens falsas e treinando o discriminador
                fake_images = generator(real_images)
                outputs = discriminator(fake_images.detach())
                # Aplica uma media para reduzir as dimensoes do output
                outputs = torch.mean(outputs, dim=[2, 3])
                loss_fake = criterion(outputs, fake_labels)
                
                # Perda total do discriminador
                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_d.step()
                
                # Atualizando o gerador
                optimizer_g.zero_grad()
                fake_images = generator(real_images)
                outputs = discriminator(fake_images)
                
                # Perda adversarial e de reconstrução
                outputs = torch.mean(outputs, dim=[2, 3])
                loss_g_adv = criterion(outputs, real_labels)
                loss_g_recon = reconstruction_loss(fake_images, real_images)
                loss_g = loss_g_adv + 0.001 * loss_g_recon  # Soma ponderada das perdas
                loss_g.backward()
                optimizer_g.step()

                print(f'Epoch [{epoch}/{num_epochs}], loss_d: {loss_d.item():.4f}, loss_g: {loss_g.item():.4f}, samples processed: {num_samples_processed} of {len(subset)}')
            
            print("-----------------------------------------------------------------------------")
            
            # if epoch % 10 == 0:
            print(f"Saving checkpoint for epoch {epoch}, stage {stage + 1}")
            psnr_value = metrics.calculate_psnr(fake_images, real_images)
            ssim_value = metrics.calculate_ssim(fake_images, real_images)
            state_handlers.save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, psnr_value, ssim_value, f'checkpoint_epoch_{epoch}_stage_{stage + 1}.pth')
            vutils.save_image(fake_images, f'artifacts/fake_images_epoch_{epoch}_stage_{stage + 1}.png')

