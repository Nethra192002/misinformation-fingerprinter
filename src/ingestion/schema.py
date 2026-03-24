from dataclasses import dataclass, field
from typing import Optional
import uuid

@dataclass
class ClaimRecord:
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    canonical_text: str = ""
    original_text: str = ""
    platform: str = ""
    source_dataset: str = ""
    timestamp: Optional[str] = None
    url: Optional[str] = None
    thread_id: Optional[str] = None
    parent_id: Optional[str] = None
    veracity_label: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "claim_id": self.claim_id,
            "canonical_text": self.canonical_text,
            "original_text": self.original_text,
            "platform": self.platform,
            "source_dataset": self.source_dataset,
            "timestamp": self.timestamp,
            "url": self.url,
            "thread_id": self.thread_id,
            "parent_id": self.parent_id,
            "veracity_label": self.veracity_label,
            "metadata": self.metadata
        }