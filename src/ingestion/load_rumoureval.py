import os
import json
import pandas as pd
from schema import ClaimRecord

def load_rumoureval(rumoureval_root: str) -> list[ClaimRecord]:
    records = []

    splits = [
        ("rumoureval-2019-training-data", "train-key.json"),
        ("rumoureval-2019-training-data", "dev-key.json"),
    ]

    for split_folder, key_file in splits:
        split_path = os.path.join(rumoureval_root, split_folder)
        key_path = os.path.join(split_path, key_file)

        if not os.path.exists(key_path):
            print(f"Missing key file: {key_path}")
            continue

        with open(key_path, "r", encoding="utf-8") as f:
            keys = json.load(f)

        veracity_map = keys.get("subtaskbenglish", {})
        stance_map = keys.get("subtaskaenglish", {})

        for subfolder in ["twitter-english", "reddit-training-data", "reddit-dev-data"]:
            data_path = os.path.join(split_path, subfolder)
            if not os.path.isdir(data_path):
                continue

            platform = "reddit" if "reddit" in subfolder else "twitter"
            event_folders = _get_event_folders(data_path, platform)

            for event_name, thread_path, thread_id in event_folders:
                source = _load_source(thread_path, thread_id, platform, event_name, veracity_map)
                if source:
                    records.append(source)

                replies = _load_replies(thread_path, thread_id, platform, event_name, stance_map)
                records.extend(replies)

    test_records = _load_test_split(rumoureval_root)
    records.extend(test_records)

    return records


def _get_event_folders(data_path: str, platform: str):
    results = []

    if platform == "twitter":
        for event_name in os.listdir(data_path):
            event_path = os.path.join(data_path, event_name)
            if not os.path.isdir(event_path) or event_name.startswith("."):
                continue
            for thread_id in os.listdir(event_path):
                thread_path = os.path.join(event_path, thread_id)
                if os.path.isdir(thread_path) and not thread_id.startswith("."):
                    results.append((event_name, thread_path, thread_id))
    else:
        for thread_id in os.listdir(data_path):
            thread_path = os.path.join(data_path, thread_id)
            if os.path.isdir(thread_path) and not thread_id.startswith("."):
                results.append(("reddit", thread_path, thread_id))

    return results


def _load_source(thread_path, thread_id, platform, event_name, veracity_map):
    source_dir = os.path.join(thread_path, "source-tweet")
    if not os.path.isdir(source_dir):
        return None

    for fname in os.listdir(source_dir):
        if not fname.endswith(".json") or fname.startswith("."):
            continue
        with open(os.path.join(source_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        text = _extract_text(data, platform)
        if not text:
            return None

        return ClaimRecord(
            original_text=text,
            canonical_text=text,
            platform=platform,
            source_dataset="rumoureval",
            timestamp=data.get("created_at") or data.get("created_utc", ""),
            thread_id=thread_id,
            parent_id=None,
            veracity_label=veracity_map.get(thread_id),
            metadata={"event": event_name, "post_id": str(data.get("id", ""))}
        )
    return None


def _load_replies(thread_path, thread_id, platform, event_name, stance_map):
    records = []
    replies_dir = os.path.join(thread_path, "replies")
    if not os.path.isdir(replies_dir):
        return records

    for fname in os.listdir(replies_dir):
        if not fname.endswith(".json") or fname.startswith("."):
            continue
        with open(os.path.join(replies_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        text = _extract_text(data, platform)
        if not text:
            continue

        reply_id = str(data.get("id", fname.replace(".json", "")))

        records.append(ClaimRecord(
            original_text=text,
            canonical_text=text,
            platform=platform,
            source_dataset="rumoureval",
            timestamp=data.get("created_at") or str(data.get("created_utc", "")),
            thread_id=thread_id,
            parent_id=thread_id,
            veracity_label=stance_map.get(reply_id),
            metadata={"event": event_name, "post_id": reply_id}
        ))

    return records


def _load_test_split(rumoureval_root: str) -> list[ClaimRecord]:
    records = []
    test_root = os.path.join(rumoureval_root, "rumoureval-2019-test-data")
    if not os.path.isdir(test_root):
        return records

    for subfolder in ["twitter-en-test-data", "reddit-test-data"]:
        data_path = os.path.join(test_root, subfolder)
        if not os.path.isdir(data_path):
            continue

        platform = "reddit" if "reddit" in subfolder else "twitter"
        event_folders = _get_event_folders(data_path, platform)

        for event_name, thread_path, thread_id in event_folders:
            source = _load_source(thread_path, thread_id, platform, event_name, {})
            if source:
                records.append(source)
            replies = _load_replies(thread_path, thread_id, platform, event_name, {})
            records.extend(replies)

    return records

def _extract_text(data: dict, platform: str) -> str:
    if platform == "twitter":
        return data.get("text", "").strip()
    else:
        if isinstance(data.get("data"), dict) and "children" in data["data"]:
            try:
                post = data["data"]["children"][0]["data"]
                title = post.get("title", "").strip()
                selftext = post.get("selftext", "").strip()
                return (title + " " + selftext).strip() if selftext else title
            except (IndexError, KeyError, TypeError):
                pass
        title = data.get("title", "").strip()
        selftext = data.get("body", data.get("selftext", "")).strip()
        return (title + " " + selftext).strip() if selftext else title
    
if __name__ == "__main__":
    records = load_rumoureval("../../data/raw/rumoureval")
    df = pd.DataFrame([r.to_dict() for r in records])
    df.to_csv("../../data/processed/rumoureval_processed.csv", index=False)
    print(f"Loaded {len(df)} records from RumourEval")
    print(f"Platforms: {df['platform'].value_counts().to_dict()}")
    print(f"Source posts: {df['parent_id'].isna().sum()}")
    print(f"Replies: {df['parent_id'].notna().sum()}")
    print(f"Veracity labels: {df['veracity_label'].value_counts().to_dict()}")