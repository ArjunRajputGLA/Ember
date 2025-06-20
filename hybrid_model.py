import torch
import torch.nn as nn
from transformers import AutoModel
import timm

class TransformerHybridModel(nn.Module):
    def __init__(self, image_size=64, text_embedding_dim=770, hidden_dim=384, num_classes=2, num_families=25):
        super(TransformerHybridModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Vision Transformer for images
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.vit.patch_embed.proj = nn.Conv2d(1, 192, kernel_size=16, stride=16, padding=0)
        self.vit_output_dim = 192
        self.vit_projection = nn.Linear(self.vit_output_dim, hidden_dim)

        # BERT for text
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Linear(text_embedding_dim, hidden_dim)

        # Cross-attention fusion
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)

        # Classifier (binary and family)
        self.binary_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        self.family_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_families)
        )

    def forward(self, image, text_inputs):
        # Image branch
        image = torch.nn.functional.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image_features = self.vit(image)
        image_features = self.vit_projection(image_features)

        # Text branch
        text_outputs = self.text_encoder(**text_inputs)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)

        # Cross-attention fusion
        image_features = image_features.unsqueeze(0)
        text_features = text_features.unsqueeze(0)
        combined_features, _ = self.cross_attention(image_features, text_features, text_features)
        combined_features = combined_features.squeeze(0)

        # Concatenate for classifier
        fused_features = torch.cat((image_features.squeeze(0), text_features.squeeze(0)), dim=1)

        # Classifiers
        binary_output = self.binary_classifier(fused_features)
        family_output = self.family_classifier(fused_features)
        return binary_output, family_output