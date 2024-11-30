import torchmetrics
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torchmetrics.image

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    
    max_pixel_value = 1.0
    psnr = 10 * torch.log10(max_pixel_value ** 2 / mse)
    return psnr.item()

def calculate_ssim(img1, img2):
    ssim_fn  = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_value = ssim_fn(img1, img2)
    return ssim_value.item()

def calculate_jaccard(img1, img2, threshold=0.5):
    img1 = (img1 > threshold).float()
    img2 = (img2 > threshold).float()

    intersection = torch.sum(img1 * img2)
    union = torch.sum(img1) + torch.sum(img2) - intersection

    jaccard = intersection / union if union != 0 else 0
    return jaccard.item()


def calculate_pearson(img1, img2):
    assert img1.shape == img2.shape, "As imagens devem ter o mesmo tamanho"
    
    img1 = img1.flatten()
    img2 = img2.flatten()
    
    mean1, mean2 = torch.mean(img1), torch.mean(img2)
    std1, std2 = torch.std(img1), torch.std(img2)
    
    covariance = torch.mean((img1 - mean1) * (img2 - mean2))
    pearson = covariance / (std1 * std2) if std1 != 0 and std2 != 0 else 0
    return pearson.item()

def calculate_mse(img1, img2):    
    return F.mse_loss(img1, img2).item()

def calculate_entropy(img):
    img = (img * 255).to(torch.int32)

    histogram = torch.histc(img.float(), bins=256, min=0, max=255)
    
    prob = histogram / histogram.sum()
    
    prob = prob[prob > 0]

    entropy = -torch.sum(prob * torch.log2(prob))
    return entropy.item()