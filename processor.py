import torch
import state_handlers
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import metrics
import random
import pandas as pd
from PIL import Image
from discriminator import Discriminator
from generator import Generator

def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5
    return tensor

IMAGE_TO_COMPRESS_PATH = "dataset/all_images/"
randons = []
models = ["artifacts/1024/checkpoint_epoch_100.pth", "artifacts/checkpoint_epoch_19_stage_5.pth"]

generator = Generator()
discriminator = Discriminator()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

while len(randons) < 1000:
    rand = random.randint(1, 10240)
    if rand not in randons:
        randons.append(rand)


for j in range(len(models)):
    state_handlers.load_checkpoint(generator, discriminator, optimizer_g, optimizer_d, models[j])
    generator.eval()
    result_data = []

    for i in range(len(randons)):
        img = Image.open(f"{IMAGE_TO_COMPRESS_PATH}/{randons[i]:06}.jpg")

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
        img_tensor_pil = transform_to_pil(denormalize(img_tensor.squeeze(0)))

        psnr = metrics.calculate_psnr(img_tensor, compressed_image)
        ssim = metrics.calculate_ssim(img_tensor, compressed_image)
        jaccard = metrics.calculate_jaccard(img_tensor, compressed_image)
        pearson = metrics.calculate_pearson(img_tensor, compressed_image)
        mse = metrics.calculate_mse(img_tensor, compressed_image)
        entropy = metrics.calculate_entropy(img_tensor)

        result_data.append({
            "Image": f"{randons[i]:06}.jpg", 
            "PSNR": psnr, 
            "SSIM": ssim, 
            "Jaccard": jaccard, 
            "Pearson": pearson, 
            "MSE": mse, 
            "Entropy": entropy
            })
                
    df = pd.DataFrame(result_data)

    print(df)

    df.to_csv(f'results_{j}.csv', index=False)