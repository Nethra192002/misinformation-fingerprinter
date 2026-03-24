import os
import pandas as pd
from schema import ClaimRecord

def load_fakenewsnet(fnn_root: str) -> list[ClaimRecord]:
    records = []

    files = [
        ("BuzzFeed_fake_news_content.csv", "buzzfeed", "fake"),
        ("BuzzFeed_real_news_content.csv", "buzzfeed", "real"),
        ("PolitiFact_fake_news_content.csv", "politifact", "real"),
        ("PolitiFact_real_news_content.csv", "politifact", "real"),
    ]

    for filename, source, default_label in files:
        fpath = os.path.join(fnn_root, filename)
        if not os.path.exists(fpath):
            print(f"Missing: {filename}")
            continue

        df = pd.read_csv(fpath)
        df = df.dropna(subset=["title"])

        for _, row in df.iterrows():
            raw_id = str(row.get("id", ""))
            veracity = _parse_veracity(raw_id, default_label)

            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()
            canonical_text = title if title else text[:300]

            record = ClaimRecord(
                original_text=title,
                canonical_text=canonical_text,
                platform="news",
                source_dataset=f"fakenewsnet_{source}",
                timestamp=str(row.get("publish_date", "")),
                url=str(row.get("url", "")),
                thread_id=raw_id,
                parent_id=None,
                veracity_label=veracity,
                metadata={
                    "source": source,
                    "authors": str(row.get("authors", "")),
                    "full_text_preview": text[:200] if text else ""
                }
            )
            records.append(record)

    return records


def _parse_veracity(raw_id: str, default_label: str) -> str:
    if raw_id.lower().startswith("real"):
        return "real"
    elif raw_id.lower().startswith("fake"):
        return "fake"
    return default_label


if __name__ == "__main__":
    records = load_fakenewsnet("../../data/raw/fakenewsnet")
    df = pd.DataFrame([r.to_dict() for r in records])
    df.to_csv("../../data/processed/fakenewsnet_processed.csv", index=False)
    print(f"Loaded {len(df)} records from FakeNewsNet")
    print(f"Sources: {df['source_dataset'].value_counts().to_dict()}")
    print(f"Veracity: {df['veracity_label'].value_counts().to_dict()}")
    print(f"Missing text: {df['original_text'].isna().sum()}")