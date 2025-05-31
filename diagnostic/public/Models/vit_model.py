import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

# Paramètres du modèle
image_size = 224  # Taille des images (224x224)
patch_size = 16   # Taille des patches (16x16)
num_patches = (image_size // patch_size) ** 2  # Nombre de patches par image
projection_dim = 64  # Dimension des embeddings
num_heads = 8        # Nombre de têtes pour l'attention multi-head
transformer_layers = 6  # Nombre de couches Transformer
mlp_head_units = [2048, 1024]  # Unités pour le MLP de classification
num_classes = 1  # Classification binaire (0: bénignes, 1: malignes)
batch_size = 32
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Chargement des données
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = datasets.ImageFolder(data_dir, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def load_data(data_dir):
    # Transformations pour l'ensemble d'entraînement (avec augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformations pour l'ensemble de validation (sans augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Chargement des datasets
    full_dataset = datasets.ImageFolder(data_dir)

    # Séparation train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Application des transformations
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Création des DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# 2. Création du modèle ViT avec PyTorch
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # Couche qui fait la projection des patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)

        # Ajout du position embedding
        x = x + self.pos_embed

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calcul des scores d'attention
        scale = (self.head_dim) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        # Application des scores aux valeurs
        x = (attn @ v).transpose(1, 2).reshape(B, N, E)
        x = self.proj(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention
        x = x + self.dropout(self.attn(self.norm1(x)))
        # MLP
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=image_size, patch_size=patch_size, embed_dim=projection_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads) for _ in range(transformer_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(projection_dim)
        self.head = nn.Sequential(
            nn.Linear(projection_dim, mlp_head_units[0]),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_head_units[0], mlp_head_units[1]),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_head_units[1], num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification head
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)

        return x