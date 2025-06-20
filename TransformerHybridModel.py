import torch
import torch.nn as nn
import timm
from transformers import AutoModel

class TransformerHybridModel(nn.Module):
    def __init__(self, vit_model='vit_small_patch16_224', text_embedding_dim=770, hidden_dim=384, num_classes=2, num_years=5):
        super(TransformerHybridModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Vision Transformer for images
        self.vit = timm.create_model(vit_model, pretrained=True, num_classes=0)
        self.vit.patch_embed.proj = nn.Conv2d(1, self.vit.embed_dim, kernel_size=16, stride=16, padding=0)
        self.vit_output_dim = self.vit.embed_dim
        self.vit_projection = nn.Linear(self.vit_output_dim, hidden_dim)

        # Text branch (using precomputed embeddings)
        self.text_projection = nn.Linear(text_embedding_dim, hidden_dim)

        # Cross-attention fusion with Transformer encoder
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1),
            num_layers=2
        )

        # Classifier (binary classification)
        self.binary_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        # Auxiliary task: Predict 'appeared' year
        self.year_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_years)  # Assuming years 2016-2020 (5 classes)
        )

    def forward(self, image, text_features):
        # Image branch
        image = torch.nn.functional.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image_features = self.vit(image)
        image_features = self.vit_projection(image_features)

        # Text branch
        text_features = self.text_projection(text_features)

        # Cross-attention fusion
        image_features = image_features.unsqueeze(0)  # (1, batch_size, hidden_dim)
        text_features = text_features.unsqueeze(0)    # (1, batch_size, hidden_dim)
        combined_features, _ = self.cross_attention(image_features, text_features, text_features)
        combined_features = self.fusion_transformer(combined_features)  # (1, batch_size, hidden_dim)
        combined_features = combined_features.squeeze(0)  # (batch_size, hidden_dim)

        # Concatenate for classifier
        fused_features = torch.cat((image_features.squeeze(0), text_features.squeeze(0)), dim=1)

        # Classifiers
        binary_output = self.binary_classifier(fused_features)
        year_output = self.year_classifier(fused_features)
        return binary_output, year_output