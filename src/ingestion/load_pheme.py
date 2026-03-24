import os
import json
import pandas as pd
from tqdm import tqdm
from schema import ClaimRecord

def load_pheme(pheme_root: str) -> list[ClaimRecord]:
    records = []
    annotated_root = os.path.join(pheme_root, "all-rnr-annotated-threads")

    for event_folder in os.listdir(annotated_root):
        event_path = os.path.join(annotated_root, event_folder)
        if not os.path.isdir(event_path) or event_folder.startswith("."):
            continue

        event_name = event_folder.replace("-all-rnr-threads", "")

        for veracity_label in ["rumours", "non-rumours"]:
            veracity_path = os.path.join(event_path, veracity_label)
            if not os.path.isdir(veracity_path):
                continue

            for thread_id in os.listdir(veracity_path):
                thread_path = os.path.join(veracity_path, thread_id)
                if not os.path.isdir(thread_path) or thread_id.startswith("."):
                    continue

                source = _load_source_tweet(thread_path, thread_id, event_name, veracity_label)
                if source:
                    records.append(source)

                reactions = _load_reactions(thread_path, thread_id, event_name)
                records.extend(reactions)

    return records


def _load_source_tweet(thread_path: str, thread_id: str, event_name: str, veracity_label: str) -> ClaimRecord | None:
    source_dir = os.path.join(thread_path, "source-tweets")
    if not os.path.isdir(source_dir):
        return None

    for fname in os.listdir(source_dir):
        if not fname.endswith(".json") or fname.startswith("."):
            continue
        fpath = os.path.join(source_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            tweet = json.load(f)

        text = tweet.get("text", "").strip()
        if not text:
            return None

        veracity = _load_veracity(thread_path)

        return ClaimRecord(
            original_text=text,
            canonical_text=text,
            platform="twitter",
            source_dataset="pheme",
            timestamp=tweet.get("created_at"),
            thread_id=thread_id,
            parent_id=None,
            veracity_label=veracity if veracity else veracity_label,
            metadata={"event": event_name, "tweet_id": tweet.get("id_str")}
        )
    return None


def _load_reactions(thread_path: str, thread_id: str, event_name: str) -> list[ClaimRecord]:
    reactions_dir = os.path.join(thread_path, "reactions")
    records = []

    if not os.path.isdir(reactions_dir):
        return records

    for fname in os.listdir(reactions_dir):
        if not fname.endswith(".json") or fname.startswith("."):
            continue
        fpath = os.path.join(reactions_dir, fname)

        with open(fpath, "r", encoding="utf-8") as f:
            tweet = json.load(f)

        text = tweet.get("text", "").strip()
        if not text:
            continue

        records.append(ClaimRecord(
            original_text=text,
            canonical_text=text,
            platform="twitter",
            source_dataset="pheme",
            timestamp=tweet.get("created_at"),
            thread_id=thread_id,
            parent_id=thread_id,
            veracity_label=None,
            metadata={
                "event": event_name,
                "tweet_id": tweet.get("id_str"),
                "in_reply_to": tweet.get("in_reply_to_status_id_str")
            }
        ))

    return records


def _load_veracity(thread_path: str) -> str | None:
    annotation_path = os.path.join(thread_path, "annotation.json")
    if not os.path.exists(annotation_path):
        return None
    with open(annotation_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    return ann.get("veracity")


if __name__ == "__main__":
    records = load_pheme("../../data/raw/pheme")
    df = pd.DataFrame([r.to_dict() for r in records])
    df.to_csv("../../data/processed/pheme_processed.csv", index=False)
    print(f"Loaded {len(df)} records from PHEME")
    print(f"Events: {df['metadata'].apply(lambda x: eval(x)['event'] if isinstance(x, str) else x.get('event')).unique()}")
    print(f"Source tweets: {df['parent_id'].isna().sum()}")
    print(f"Reactions: {df['parent_id'].notna().sum()}")
    print(f"Veracity labels: {df['veracity_label'].value_counts().to_dict()}")