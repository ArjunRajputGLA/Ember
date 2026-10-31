import os
import json
import time

test_path = r"c:\Users\imsto\Desktop\Final Year Paper\ember_dataset_2018_2\ember2018\test_features.jsonl"
print(f"Checking test file: {test_path}")
print(f"File exists: {os.path.exists(test_path)}")
print(f"File size: {os.path.getsize(test_path) / (1024**3):.2f} GB")

t0 = time.time()
with open(test_path, "r", encoding="utf-8") as f:
    first_line = f.readline()
    rec = json.loads(first_line)
    print("First record keys:", list(rec.keys()))
    print("First record label:", rec.get("label"))
    print("First record sha256:", rec.get("sha256"))
    print(f"Check took: {time.time() - t0:.4f}s")

