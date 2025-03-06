import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import wandb
import pywt
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ViTModel, CLIPProcessor, CLIPModel

# Initialize Weights & Biases
wandb.init(project="wavelet-vit-latent-diffusion", name="imagenet-experiment")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP Model for Interpretability Analysis
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 1. Wavelet Transform-based Latent Feature Extractor
class WaveletLatentFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        coeffs = pywt.wavedec2(x.cpu().numpy(), 'haar', level=2)
        low_freq, details = coeffs[0], coeffs[1:]
        return torch.tensor(low_freq).to(device), details

# 2. Vision Transformer for Hierarchical Latent Refinement
class HierarchicalLatentViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.fc = nn.Linear(768, 512)
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.vit(inputs_embeds=x).last_hidden_state.mean(dim=1)
        return self.fc(x)

# 3. Wavelet-Based Latent Diffusion Model
class WaveletLatentDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = WaveletLatentFeatureExtractor()
        self.transformer = HierarchicalLatentViT()
        self.denoise = nn.Linear(512, 512)
    
    def forward(self, x, noise_level):
        low_freq, details = self.feature_extractor(x)
        refined_features = self.transformer(low_freq)
        refined_features = self.denoise(refined_features + noise_level * torch.randn_like(refined_features))
        return refined_features, details

# 4. Load High-Resolution Image Dataset (ImageNet)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
dataset = torchvision.datasets.ImageNet(root="./data", split='train', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 5. Training Loop
model = WaveletLatentDiffusion().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5
noise_level = 0.1

for epoch in range(epochs):
    epoch_loss = 0
    for images, _ in tqdm(dataloader):
        images = images.to(device)
        optimizer.zero_grad()
        noisy_images = images + noise_level * torch.randn_like(images)
        low_refined, details = model(noisy_images, noise_level)
        loss = F.mse_loss(low_refined, images.view(images.shape[0], -1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # CLIP-Based Interpretability Analysis
        image_features = clip_model.get_image_features(images)
        text_features = clip_model.get_text_features(clip_processor(["a high-resolution image"], return_tensors="pt")["input_ids"].to(device))
        similarity = torch.cosine_similarity(image_features, text_features).mean().item()
        
        # Log intermediate visualizations to wandb
        wandb.log({
            "epoch": epoch, 
            "loss": epoch_loss / len(dataloader),
            "Low-Frequency Features": [wandb.Image(low_refined.cpu().numpy(), caption="Low Level")],
            "Detail Coefficients": [wandb.Image(details[0][0].cpu().numpy(), caption="Details")],
            "CLIP Similarity": similarity
        })

print("Training Complete!")

# 6. Explainability - Visualizing Wavelet Transform & Hierarchical Features
def visualize_wavelet_features(model, image):
    model.eval()
    with torch.no_grad():
        low_freq, details = model.feature_extractor(image.unsqueeze(0).to(device))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[1].imshow(low_freq.cpu().numpy(), cmap="gray")
    axes[1].set_title("Low-Frequency Features")
    axes[2].imshow(details[0][0].cpu().numpy(), cmap="gray")
    axes[2].set_title("Detail Coefficients")
    plt.show()

# Sample Visualization
sample_image, _ = dataset[0]
visualize_wavelet_features(model, sample_image)

# 7. Log Feature Maps to wandb
def log_to_wandb(model, image):
    model.eval()
    with torch.no_grad():
        low_freq, details = model.feature_extractor(image.unsqueeze(0).to(device))
        image_features = clip_model.get_image_features(image.unsqueeze(0).to(device))
        text_features = clip_model.get_text_features(clip_processor(["a high-resolution image"], return_tensors="pt")["input_ids"].to(device))
        similarity = torch.cosine_similarity(image_features, text_features).mean().item()
    
    wandb.log({
        "Original Image": [wandb.Image(image.permute(1, 2, 0).cpu().numpy(), caption="Original")],
        "Low-Frequency Features": [wandb.Image(low_freq.cpu().numpy(), caption="Low Level")],
        "Detail Coefficients": [wandb.Image(details[0][0].cpu().numpy(), caption="Details")],
        "CLIP Similarity": similarity
    })

# Log First Sample
log_to_wandb(model, sample_image)
