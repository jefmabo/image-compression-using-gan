import torch
import state_handlers
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import cv2
import torchvision.transforms as transforms
import metrics
from PIL import Image
from discriminator import Discriminator
from generator import Generator

def denormalize(tensor):
    # Reverte a normalização aplicada, retornando o intervalo para [0, 1]
    tensor = tensor * 0.5 + 0.5
    return tensor

def compress(image_path):

    generator = Generator()
    discriminator = Discriminator()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    epoch, psnr, ssim = state_handlers.load_checkpoint(generator, discriminator, optimizer_g, optimizer_d, "artifacts/checkpoint_epoch_100.pth")

    generator.eval()

    img = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize((160, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        compressed_image = generator(img_tensor)

    # Desnormalizar a imagem para exibição
    compressed_image_denorm = denormalize(compressed_image.squeeze(0))  # Remove a dimensão do batch

    # Converter o tensor comprimido de volta para imagem PIL e exibir
    transform_to_pil = transforms.ToPILImage()
    img_compressed_pil = transform_to_pil(compressed_image_denorm)

    # Exibir as duas imagens (original e comprimida) lado a lado
    plt.figure(figsize=(10, 5))

    # Exibir a imagem original
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Imagem Original')
    plt.axis('off')

    # Exibir a imagem comprimida
    plt.subplot(1, 2, 2)
    plt.imshow(img_compressed_pil)
    plt.title('Imagem Comprimida')
    plt.axis('off')

    # Mostrar as imagens
    plt.show()