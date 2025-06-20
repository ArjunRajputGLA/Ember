import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel
from TransformerHybridModel import TransformerHybridModel
from scipy.stats import entropy
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
image_dir = "data/processed/malimg/"
nlp_dir = "data/processed/nlp_vectors/"
os.makedirs(nlp_dir, exist_ok=True)

# Use CPU
device = torch.device("cpu")
logging.info(f"Using device: {device} (No GPU detected)")

# Load data in chunks
logging.info("Loading data...")
chunk_size = 10000
train_images_chunks = [np.loadtxt(os.path.join(image_dir, "malimg_train_images.csv"), delimiter=",")[i:i + chunk_size] for i in range(0, 120000, chunk_size)]
test_images_chunks = [np.loadtxt(os.path.join(image_dir, "malimg_test_images.csv"), delimiter=",")[i:i + chunk_size] for i in range(0, 40000, chunk_size)]
train_metadata = pd.read_csv("data/processed/subsampled/train_vectors_sub.csv")
test_metadata = pd.read_csv("data/processed/subsampled/test_vectors_sub.csv")

# Preprocess textual features
def preprocess_text(row):
    try:
        text = " ".join([str(eval(row['imports']) or []), str(eval(row['exports']) or []), str(eval(row['section']) or [])]).lower().strip()
        return text if text else "unknown"
    except Exception as e:
        logging.warning(f"Error processing row: {e}")
        return "unknown"

logging.info("Preprocessing textual features...")
train_texts = train_metadata.apply(preprocess_text, axis=1)
test_texts = test_metadata.apply(preprocess_text, axis=1)

# Extract static analysis features
logging.info("Extracting static analysis features...")
train_metadata['section_sizes'] = train_metadata['section'].apply(lambda x: sum([len(s) for s in eval(x)]) if isinstance(x, str) else 0)
test_metadata['section_sizes'] = test_metadata['section'].apply(lambda x: sum([len(s) for s in eval(x)]) if isinstance(x, str) else 0)
train_metadata['entropy'] = train_metadata['histogram'].apply(lambda x: entropy(eval(x)) if isinstance(x, str) else 0)
test_metadata['entropy'] = test_metadata['histogram'].apply(lambda x: entropy(eval(x)) if isinstance(x, str) else 0)

train_section_sizes = np.array(train_metadata['section_sizes'].values).reshape(-1, 1)
test_section_sizes = np.array(test_metadata['section_sizes'].values).reshape(-1, 1)
train_entropy = np.array(train_metadata['entropy'].values).reshape(-1, 1)
test_entropy = np.array(test_metadata['entropy'].values).reshape(-1, 1)

# Extract year labels for auxiliary task
train_metadata['year'] = train_metadata['appeared'].apply(lambda x: int(x.split('-')[0]))
test_metadata['year'] = test_metadata['appeared'].apply(lambda x: int(x.split('-')[0]))
year_mapping = {year: idx for idx, year in enumerate(sorted(train_metadata['year'].unique()))}
train_years = train_metadata['year'].map(year_mapping).values
test_years = test_metadata['year'].map(year_mapping).values
num_years = len(year_mapping)
logging.info(f"Number of unique years: {num_years}, Mapping: {year_mapping}")

# Tokenize texts
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_text_inputs = tokenizer(train_texts.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=64)
test_text_inputs = tokenizer(test_texts.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=64)

# Load or compute BERT embeddings
train_embeddings_path = os.path.join(nlp_dir, "train_text_embeddings.pt")
test_embeddings_path = os.path.join(nlp_dir, "test_text_embeddings.pt")

def compute_bert_embeddings(text_inputs, section_sizes, entropy_vals, batch_size=128):
    embeddings = []
    total_samples = len(text_inputs['input_ids'])
    total_batches = (total_samples + batch_size - 1) // batch_size
    logging.info(f"Total samples: {total_samples}, Total batches: {total_batches}")

    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    section_sizes_tensor = torch.tensor(section_sizes, dtype=torch.float32).to(device)
    entropy_tensor = torch.tensor(entropy_vals, dtype=torch.float32).to(device)

    bert_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
    bert_model.eval()

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, total_samples)
            batch_inputs = {k: v[batch_start:batch_end] for k, v in text_inputs.items()}
            batch_outputs = bert_model(**batch_inputs)
            batch_embeddings = batch_outputs.last_hidden_state[:, 0, :].to(torch.float16)  # Use float16

            batch_section_sizes = section_sizes_tensor[batch_start:batch_end]
            batch_entropy = entropy_tensor[batch_start:batch_end]
            batch_combined = torch.cat((batch_embeddings, batch_section_sizes, batch_entropy), dim=1)
            embeddings.append(batch_combined)

            if (i // batch_size) % 50 == 0:
                logging.info(f"Processed batch {(i // batch_size) + 1}/{total_batches}")

    return torch.cat(embeddings, dim=0)

if os.path.exists(train_embeddings_path) and os.path.exists(test_embeddings_path):
    logging.info("Loading precomputed BERT embeddings from disk...")
    train_text_embeddings = torch.load(train_embeddings_path, map_location=device).to(torch.float16)
    test_text_embeddings = torch.load(test_embeddings_path, map_location=device).to(torch.float16)
else:
    logging.info("Computing BERT embeddings for training set...")
    train_text_embeddings = compute_bert_embeddings(train_text_inputs, train_section_sizes, train_entropy)
    logging.info("Computing BERT embeddings for test set...")
    test_text_embeddings = compute_bert_embeddings(test_text_inputs, test_section_sizes, test_entropy)
    logging.info("Saving BERT embeddings to disk...")
    torch.save(train_text_embeddings, train_embeddings_path)
    torch.save(test_text_embeddings, test_embeddings_path)

# Labels
train_labels = train_metadata['label'].values
test_labels = test_metadata['label'].values

# Normalize and convert to tensors with lower precision
train_images = [chunk / 255.0 for chunk in train_images_chunks]
test_images = [chunk / 255.0 for chunk in test_images_chunks]
train_images = [torch.tensor(chunk, dtype=torch.float16).reshape(-1, 1, 64, 64) for chunk in train_images]
test_images = [torch.tensor(chunk, dtype=torch.float16).reshape(-1, 1, 64, 64) for chunk in test_images_chunks]
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)
train_years = torch.tensor(train_years, dtype=torch.long)
test_years = torch.tensor(test_years, dtype=torch.long)

# Create dataset in chunks
train_datasets = [TensorDataset(img_chunk, train_text_embeddings[i:i + chunk_size], train_labels[i:i + chunk_size], train_years[i:i + chunk_size]) for i, img_chunk in enumerate(train_images)]
test_dataset = TensorDataset(test_images[0], test_text_embeddings, test_labels, test_years)  # Assuming single chunk for test

train_loader = [DataLoader(ds, batch_size=64, shuffle=True, num_workers=0) for ds in train_datasets]
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# Compute class weights
num_malicious = (train_labels == 1).sum().item()
num_benign = (train_labels == 0).sum().item()
total_samples = len(train_labels)
weight_benign = total_samples / (2 * num_benign)
weight_malicious = total_samples / (2 * num_malicious)
class_weights = torch.tensor([weight_benign, weight_malicious], dtype=torch.float32).to(device)
logging.info(f"Class weights - Benign: {weight_benign:.4f}, Malicious: {weight_malicious:.4f}")

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()

# Label Smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    def forward(self, inputs, targets):
        confidence = 1.0 - self.smoothing
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

# MixUp
def mixup_data(x1, x2, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    x_mixed = lam * x1 + (1 - lam) * x2
    y_mixed = lam * y + (1 - lam) * y
    return x_mixed, y_mixed

# Model, optimizer, and loss
model = TransformerHybridModel(vit_model='vit_tiny_patch16_224', num_years=num_years).to(device)
optimizer = optim.Adam([
    {'params': [p for n, p in model.named_parameters() if 'text_encoder' not in n], 'lr': 5e-4},
    {'params': [p for n, p in model.named_parameters() if 'text_encoder' in n], 'lr': 2e-5}
], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)
criterion = FocalLoss(gamma=2.0, alpha=class_weights)
smoothing_criterion = LabelSmoothingLoss(smoothing=0.1)
year_criterion = nn.CrossEntropyLoss()

# K-Fold Cross-Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_images[0])):  # Use first chunk for index
    logging.info(f"Starting Fold {fold + 1}/3")

    # Create fold-specific dataloaders (combine chunks)
    train_subsets = [torch.utils.data.Subset(ds, [i % len(ds) for i in train_idx if i < len(ds) * (idx + 1) and i >= len(ds) * idx]) for idx, ds in enumerate(train_datasets)]
    train_dataset_fold = torch.utils.data.ConcatDataset(train_subsets)
    val_subset = torch.utils.data.Subset(test_dataset, val_idx[:len(test_dataset)])  # Adjust for test size
    train_fold_loader = DataLoader(train_dataset_fold, batch_size=64, shuffle=True, num_workers=0)
    val_fold_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)

    # Training loop
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(5):
        model.train()
        train_loss = 0.0
        train_year_loss = 0.0
        num_batches = 0

        for batch_idx, (batch_img, batch_text, batch_lbl, batch_year) in enumerate(train_fold_loader):
            batch_img, batch_text, batch_lbl, batch_year = batch_img.to(device), batch_text.to(device), batch_lbl.to(device), batch_year.to(device)

            # MixUp
            batch_img_mixed, batch_lbl_mixed = mixup_data(batch_img, batch_img.flip(0), batch_lbl, alpha=1.0)

            binary_output, year_output = model(batch_img_mixed, batch_text)
            binary_loss = criterion(binary_output, batch_lbl) + 0.1 * smoothing_criterion(binary_output, batch_lbl)
            year_loss = year_criterion(year_output, batch_year)
            total_loss = binary_loss + 0.5 * year_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += binary_loss.item()
            train_year_loss += year_loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        avg_train_year_loss = train_year_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_year_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_img, batch_text, batch_lbl, batch_year in val_fold_loader:
                batch_img, batch_text, batch_lbl, batch_year = batch_img.to(device), batch_text.to(device), batch_lbl.to(device), batch_year.to(device)
                binary_output, year_output = model(batch_img, batch_text)
                binary_loss = criterion(binary_output, batch_lbl)
                year_loss = year_criterion(year_output, batch_year)
                val_loss += binary_loss.item()
                val_year_loss += year_loss.item()
                val_preds = torch.argmax(binary_output, dim=1)
                val_correct += (val_preds == batch_lbl).sum().item()
                val_total += batch_lbl.size(0)

        avg_val_loss = val_loss / len(val_fold_loader)
        avg_val_year_loss = val_year_loss / len(val_fold_loader)
        val_acc = val_correct / val_total

        logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/5, Train Loss: {avg_train_loss:.4f}, Train Year Loss: {avg_train_year_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Year Loss: {avg_val_year_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"model_fold_{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break

logging.info("Training completed")