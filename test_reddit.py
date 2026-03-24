import os
path = "data/raw/rumoureval/rumoureval-2019-training-data/reddit-training-data"
threads = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and not f.startswith(".")]
print(f"Found {len(threads)} Reddit thread folders")
print(threads[:5])