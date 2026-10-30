import json, os, re
import ember.features as ef
from sklearn.feature_extraction import FeatureHasher
import numpy as np
import lightgbm as lgb

data_dir = r'c:\Users\imsto\Desktop\Final Year Paper\ember_dataset_2018_2\ember2018'
model_path = os.path.join(data_dir, 'ember_model_2018.txt')
booster = lgb.Booster(model_file=model_path)

def extract_section_token_hash(raw_obj):
    sections = raw_obj.get('sections', [])
    general = [
        len(sections),
        sum(1 for s in sections if s.get('size', 0) == 0),
        sum(1 for s in sections if s.get('name', '') == ''),
        sum(1 for s in sections if 'MEM_READ' in s.get('props', []) and 'MEM_EXECUTE' in s.get('props', [])),
        sum(1 for s in sections if 'MEM_WRITE' in s.get('props', []))
    ]
    section_sizes = [(s.get('name', ''), s.get('size', 0)) for s in sections]
    section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
    section_entropy = [(s.get('name', ''), s.get('entropy', 0)) for s in sections]
    section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
    section_vsize = [(s.get('name', ''), s.get('vsize', 0)) for s in sections]
    section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
    entry_val = raw_obj.get('entry', '')
    # Option A: token hash [[entry_val]]
    entry_name_hashed = FeatureHasher(50, input_type="string").transform([[entry_val]]).toarray()[0]
    characteristics = [p for s in sections for p in s.get('props', []) if s.get('name', '') == entry_val]
    characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

    return np.hstack([
        general, section_sizes_hashed, section_entropy_hashed, section_vsize_hashed, entry_name_hashed,
        characteristics_hashed
    ]).astype(np.float32)

def extract_section_char_hash(raw_obj):
    sections = raw_obj.get('sections', [])
    general = [
        len(sections),
        sum(1 for s in sections if s.get('size', 0) == 0),
        sum(1 for s in sections if s.get('name', '') == ''),
        sum(1 for s in sections if 'MEM_READ' in s.get('props', []) and 'MEM_EXECUTE' in s.get('props', [])),
        sum(1 for s in sections if 'MEM_WRITE' in s.get('props', []))
    ]
    section_sizes = [(s.get('name', ''), s.get('size', 0)) for s in sections]
    section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
    section_entropy = [(s.get('name', ''), s.get('entropy', 0)) for s in sections]
    section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
    section_vsize = [(s.get('name', ''), s.get('vsize', 0)) for s in sections]
    section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
    entry_val = raw_obj.get('entry', '')
    # Option B: exact sklearn 0.20 char-by-char hash [list(entry_val)]
    entry_name_hashed = FeatureHasher(50, input_type="string").transform([list(entry_val)]).toarray()[0]
    characteristics = [p for s in sections for p in s.get('props', []) if s.get('name', '') == entry_val]
    characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

    return np.hstack([
        general, section_sizes_hashed, section_entropy_hashed, section_vsize_hashed, entry_name_hashed,
        characteristics_hashed
    ]).astype(np.float32)

# Load 100 samples and compare
records = []
with open(os.path.join(data_dir, 'train_features_0.jsonl'), 'r', encoding='utf-8') as f:
    for i in range(100):
        records.append(json.loads(f.readline()))

extractor = ef.PEFeatureExtractor(feature_version=2, print_feature_warning=False)

diffs = []
pred_diffs = []

for r in records:
    # Test token hash
    ef.SectionInfo.process_raw_features = lambda self, obj: extract_section_token_hash(obj)
    v_token = extractor.process_raw_features(r)
    p_token = booster.predict([v_token])[0]

    # Test char hash (exact sklearn 0.20)
    ef.SectionInfo.process_raw_features = lambda self, obj: extract_section_char_hash(obj)
    v_char = extractor.process_raw_features(r)
    p_char = booster.predict([v_char])[0]

    diffs.append(np.max(np.abs(v_token - v_char)))
    pred_diffs.append(abs(p_token - p_char))

print(f"Max feature difference across 100 samples: {max(diffs):.6f}")
print(f"Max model prediction difference across 100 samples: {max(pred_diffs):.10f}")

