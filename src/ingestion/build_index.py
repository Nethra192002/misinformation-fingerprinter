import os
import sqlite3
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = "../../data/processed"
INDEX_DIR = "../../data/index"
DB_PATH = os.path.join(INDEX_DIR, "claims.db")
FAISS_PATH = os.path.join(INDEX_DIR, "claims.faiss")
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 64


def load_all_processed() -> pd.DataFrame:
    dfs = []
    for fname in os.listdir(PROCESSED_DIR):
        if fname.endswith("_processed.csv"):
            fpath = os.path.join(PROCESSED_DIR, fname)
            df = pd.read_csv(fpath)
            dfs.append(df)
            print(f"Loaded {len(df)} records from {fname}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["canonical_text"])
    combined = combined[combined["canonical_text"].str.strip() != ""]
    combined = combined.drop_duplicates(subset=["canonical_text"])
    combined = combined.reset_index(drop=True)
    return combined


def create_database(df: pd.DataFrame, db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE claims (
            faiss_id    INTEGER PRIMARY KEY,
            claim_id    TEXT,
            canonical_text TEXT,
            original_text  TEXT,
            platform    TEXT,
            source_dataset TEXT,
            timestamp   TEXT,
            thread_id   TEXT,
            parent_id   TEXT,
            veracity_label TEXT,
            metadata    TEXT
        )
    """)

    rows = []
    for i, row in df.iterrows():
        rows.append((
            i,
            str(row.get("claim_id", "")),
            str(row.get("canonical_text", "")),
            str(row.get("original_text", "")),
            str(row.get("platform", "")),
            str(row.get("source_dataset", "")),
            str(row.get("timestamp", "")),
            str(row.get("thread_id", "")),
            str(row.get("parent_id", "")),
            str(row.get("veracity_label", "")),
            str(row.get("metadata", "{}"))
        ))

    cursor.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    print(f"Database created with {len(rows)} records")


def build_faiss_index(df: pd.DataFrame, faiss_path: str):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = df["canonical_text"].tolist()
    all_embeddings = []

    print(f"Encoding {len(texts)} texts in batches of {BATCH_SIZE}")
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device
        )
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings).astype("float32")
    faiss.normalize_L2(all_embeddings)

    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)

    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
    faiss.write_index(index, faiss_path)
    print(f"FAISS index built with {index.ntotal} vectors of dimension {dim}")
    return model


def verify_index(faiss_path: str, db_path: str, model: SentenceTransformer):
    print("\nVerifying index with test query...")
    index = faiss.read_index(faiss_path)
    conn = sqlite3.connect(db_path)

    test_query = "shooting attack victims killed"
    query_vec = model.encode([test_query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, 5)

    print(f"Query: '{test_query}'")
    print("Top 5 matches:")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        row = conn.execute("SELECT canonical_text, platform, source_dataset FROM claims WHERE faiss_id=?", (int(idx),)).fetchone()
        if row:
            print(f"  {rank+1}. [{row[1]}/{row[2]}] (score: {dist:.3f}) {row[0][:100]}")

    conn.close()


if __name__ == "__main__":
    print("=== Phase 1: Loading processed data ===")
    df = load_all_processed()
    print(f"\nTotal unique records: {len(df)}")
    print(f"Platforms: {df['platform'].value_counts().to_dict()}")

    print("\n=== Phase 2: Building SQLite database ===")
    create_database(df, DB_PATH)

    print("\n=== Phase 3: Building FAISS index ===")
    model = build_faiss_index(df, FAISS_PATH)

    print("\n=== Phase 4: Verification ===")
    verify_index(FAISS_PATH, DB_PATH, model)

    print("\nIndex build complete.")
    print(f"  Database: {DB_PATH}")
    print(f"  FAISS index: {FAISS_PATH}")