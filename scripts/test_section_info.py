import json, os
import ember.features as ef
from sklearn.feature_extraction import FeatureHasher
import numpy as np

data_dir = r'c:\Users\imsto\Desktop\Final Year Paper\ember_dataset_2018_2\ember2018'
sec_fe = ef.SectionInfo()

print("Testing original SectionInfo.process_raw_features on real records:")
with open(os.path.join(data_dir, 'train_features_0.jsonl'), 'r', encoding='utf-8') as f:
    for i in range(10):
        line = f.readline()
        rec = json.loads(line)
        sec_obj = rec['section']
        entry_val = sec_obj.get('entry')
        n_sec = len(sec_obj.get('sections', []))
        print(f"Sample {i}: entry={entry_val!r}, num_sections={n_sec}")
        try:
            vec = sec_fe.process_raw_features(sec_obj)
            print(f"  Original succeeded: shape={vec.shape}")
        except Exception as e:
            print(f"  Original FAILED: {type(e).__name__}: {e}")

print("\nFeatureHasher behavior investigation:")
fh = FeatureHasher(50, input_type='string')
v_list = fh.transform([['.text']]).toarray()[0]
nz_idx = np.nonzero(v_list)[0]
print(f"Nonzero indices with [['.text']]: {nz_idx}, values: {v_list[nz_idx]}")

v_chars = fh.transform([list('.text')]).toarray()[0]
nz_chars = np.nonzero(v_chars)[0]
print(f"Nonzero indices with list('.text'): {nz_chars}, values: {v_chars[nz_chars]}")

# Check feature index offsets
offsets = {}
idx = 0
for fe in ef.PEFeatureExtractor(feature_version=2, print_feature_warning=False).features:
    offsets[fe.name] = (idx, idx + fe.dim)
    idx += fe.dim
print("\nFeature index offsets in 2,381-D vector:")
for k, v in offsets.items():
    print(f"  {k:<20}: {v[0]:>5} to {v[1]:>5} (dim={v[1]-v[0]})")
