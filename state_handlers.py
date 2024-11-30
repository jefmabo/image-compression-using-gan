import torch
import json

# Caminhos onde os modelos serão salvos
checkpoint_path_generator = "generator.pth"
checkpoint_path_discriminator = "discriminator.pth"
checkpoint_path = "gan_checkpoint.pth"

# Função para salvar os modelos e otimizadores
def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, psnr_value, ssim_value, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'psnr': psnr_value,
        'ssim': ssim_value
    }, f"artifacts/{checkpoint_path}")

# Carregar os dados salvos
def load_checkpoint(generator, discriminator, optimizer_g, optimizer_d, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    # Carregar os pesos dos modelos
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Carregar o estado dos otimizadores
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    

    return (checkpoint['epoch'], checkpoint['psnr'], checkpoint['ssim'])