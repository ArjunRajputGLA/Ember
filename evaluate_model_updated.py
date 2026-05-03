import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from TransformerHybridModel import TransformerHybridModel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from scipy.stats import entropy
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
image_dir = "data/processed/malimg/"
nlp_dir = "data/processed/nlp_vectors/"

# Load test data
logging.info("Loading test data...")
test_images = np.loadtxt(os.path.join(image_dir, "malimg_test_images.csv"), delimiter=",")
test_embeddings = np.loadtxt(os.path.join(nlp_dir, "test_nlp_vectors_sub.csv"), delimiter=",")
test_df = pd.read_csv("data/processed/subsampled/test_vectors_sub.csv")
test_labels = test_df['label'].values

# Preprocess textual features
def preprocess_text(row):
    try:
        imports = " ".join(eval(row['imports'])) if isinstance(row['imports'], str) and row['imports'] else ""
        exports = " ".join(eval(row['exports'])) if isinstance(row['exports'], str) and row['exports'] else ""
        sections = " ".join(eval(row['section'])) if isinstance(row['section'], str) and row['section'] else ""
        text = f"{imports} {exports} {sections}".lower().strip()
        return text if text else "unknown"
    except Exception as e:
        logging.warning(f"Error processing row: {e}")
        return "unknown"

logging.info("Preprocessing textual features...")
test_texts = test_df.apply(preprocess_text, axis=1)

# Extract static analysis features
logging.info("Extracting static analysis features...")
test_df['section_sizes'] = test_df['section'].apply(lambda x: [len(s) for s in eval(x)] if isinstance(x, str) else [0])
test_df['entropy'] = test_df['histogram'].apply(lambda x: entropy(eval(x)) if isinstance(x, str) else 0)

test_section_sizes = np.array(test_df['section_sizes'].apply(lambda x: sum(x)).values).reshape(-1, 1)
test_entropy = np.array(test_df['entropy'].values).reshape(-1, 1)

# Tokenize texts
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
test_text_inputs = tokenizer(test_texts.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=128)

# Load BERT model to compute embeddings on-the-fly
logging.info("Loading BERT model...")
bert_model = AutoModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)
bert_model.eval()

# Function to compute BERT embeddings with batching
def compute_bert_embeddings(text_inputs, section_sizes, entropy_vals, batch_size=64):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(text_inputs['input_ids']), batch_size):
            batch_inputs = {k: v[i:i + batch_size].to(device) for k, v in text_inputs.items()}
            batch_outputs = bert_model(**batch_inputs)
            batch_embeddings = batch_outputs.last_hidden_state[:, 0, :].cpu()  # [CLS] token, 768D
            # Concatenate static features
            batch_section_sizes = torch.tensor(section_sizes[i:i + batch_size], dtype=torch.float32)
            batch_entropy = torch.tensor(entropy_vals[i:i + batch_size], dtype=torch.float32)
            batch_combined = torch.cat((batch_embeddings, batch_section_sizes, batch_entropy), dim=1)  # 768 + 1 + 1 = 770
            embeddings.append(batch_combined)
    return torch.cat(embeddings, dim=0)

logging.info("Computing BERT embeddings for test set...")
test_text_embeddings = compute_bert_embeddings(test_text_inputs, test_section_sizes, test_entropy)

# Normalize test images
test_images = test_images / 255.0

# Convert to tensors
test_images = torch.tensor(test_images, dtype=torch.float32).reshape(-1, 1, 64, 64)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_images = test_images.to(device)
test_text_embeddings = test_text_embeddings.to(device)
test_labels = test_labels.to(device)

# Number of folds and batch size
num_folds = 5
batch_size = 64
all_test_preds = []
all_test_accuracies = []

# Evaluate each fold
for fold in range(num_folds):
    logging.info(f"Evaluating model from fold {fold}...")

    # Load the model for this fold
    model = TransformerHybridModel().to(device)
    model_path = f"model_fold_{fold}.pt"
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        logging.error(f"Model file {model_path} not found. Skipping this fold.")
        continue
    model.eval()

    # Evaluate on test set
    logging.info(f"Fold {fold} - Test Set Evaluation")
    fold_test_preds = []
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            batch_images = test_images[i:i + batch_size]
            batch_text = test_text_embeddings[i:i + batch_size]
            binary_outputs = model(batch_images, batch_text)
            preds = torch.argmax(binary_outputs, dim=1)
            fold_test_preds.append(preds.cpu().numpy())
    fold_test_preds = np.concatenate(fold_test_preds)
    accuracy = (fold_test_preds == test_labels.cpu().numpy()).mean()
    logging.info(f"Fold {fold} Test Accuracy: {accuracy:.4f}")
    all_test_accuracies.append(accuracy)
    all_test_preds.append(fold_test_preds)

    # Temporal drift analysis
    logging.info(f"Fold {fold} - Temporal Drift Analysis")
    test_df['year'] = test_df['appeared'].apply(lambda x: x.split('-')[0])
    benign_indices = test_df[test_df['label'] == 0].index
    benign_2017_indices = test_df[(test_df['label'] == 0) & (test_df['year'] == '2017')].index
    benign_2018_indices = test_df[(test_df['label'] == 0) & (test_df['year'] == '2018')].index

    benign_preds = fold_test_preds[benign_indices]
    benign_labels = test_labels[benign_indices].cpu().numpy()
    benign_accuracy = (benign_preds == benign_labels).mean()
    logging.info(f"Fold {fold} Benign Accuracy: {benign_accuracy:.4f}")

    benign_2017_preds = fold_test_preds[benign_2017_indices]
    benign_2017_labels = test_labels[benign_2017_indices].cpu().numpy()
    benign_2017_accuracy = (benign_2017_preds == benign_2017_labels).mean()
    logging.info(f"Fold {fold} Benign 2017 Accuracy: {benign_2017_accuracy:.4f} ({len(benign_2017_indices)} samples)")

    benign_2018_preds = fold_test_preds[benign_2018_indices]
    benign_2018_labels = test_labels[benign_2018_indices].cpu().numpy()
    benign_2018_accuracy = (benign_2018_preds == benign_2018_labels).mean()
    logging.info(f"Fold {fold} Benign 2018 Accuracy: {benign_2018_accuracy:.4f} ({len(benign_2018_indices)} samples)")

# Aggregate Results
logging.info("Aggregated Results Across All Folds")

avg_test_accuracy = np.mean(all_test_accuracies)
std_test_accuracy = np.std(all_test_accuracies)
logging.info(f"Average Test Accuracy across {num_folds} folds: {avg_test_accuracy:.4f} (±{std_test_accuracy:.4f})")

# Ensemble prediction
all_test_preds = np.stack(all_test_preds, axis=0)
ensemble_test_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_test_preds)
ensemble_test_accuracy = (ensemble_test_preds == test_labels.cpu().numpy()).mean()
logging.info(f"Ensemble Test Accuracy (majority voting): {ensemble_test_accuracy:.4f}")

# Per-class metrics and confusion matrix
precision, recall, f1, _ = precision_recall_fscore_support(test_labels.cpu().numpy(), ensemble_test_preds, average=None)
logging.info(f"Ensemble Precision: Benign {precision[0]:.4f}, Malicious {precision[1]:.4f}")
logging.info(f"Ensemble Recall: Benign {recall[0]:.4f}, Malicious {recall[1]:.4f}")
logging.info(f"Ensemble F1: Benign {f1[0]:.4f}, Malicious {f1[1]:.4f}")
cm = confusion_matrix(test_labels.cpu().numpy(), ensemble_test_preds)
logging.info(f"Confusion Matrix:\n{cm}")

# Temporal drift analysis (ensemble)
logging.info("Temporal Drift Analysis (Ensemble)")
benign_preds = ensemble_test_preds[benign_indices]
benign_labels = test_labels[benign_indices].cpu().numpy()
benign_accuracy = (benign_preds == benign_labels).mean()
logging.info(f"Ensemble Benign Accuracy: {benign_accuracy:.4f}")

benign_2017_preds = ensemble_test_preds[benign_2017_indices]
benign_2017_labels = test_labels[benign_2017_indices].cpu().numpy()
benign_2017_accuracy = (benign_2017_preds == benign_2017_labels).mean()
logging.info(f"Ensemble Benign 2017 Accuracy: {benign_2017_accuracy:.4f} ({len(benign_2017_indices)} samples)")

benign_2018_preds = ensemble_test_preds[benign_2018_indices]
benign_2018_labels = test_labels[benign_2018_indices].cpu().numpy()
benign_2018_accuracy = (benign_2018_preds == benign_2018_labels).mean()
logging.info(f"Ensemble Benign 2018 Accuracy: {benign_2018_accuracy:.4f} ({len(benign_2018_indices)} samples)")