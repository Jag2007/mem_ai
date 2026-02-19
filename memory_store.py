
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def _term_freq(tokens: List[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens) or 1
    return {token: count / total for token, count in counts.items()}


def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0

    dot = 0.0
    for key, value in a.items():
        dot += value * b.get(key, 0.0)

    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class Memory:
    fact: str
    created_at: str
    source_user_message: str


class MemoryStore:
    def __init__(self, path: str = "memories.json"):
        self.path = Path(path)
        self.memories: List[Memory] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.memories = []
            return

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.memories = [Memory(**item) for item in raw]
        except (json.JSONDecodeError, TypeError, KeyError):
            # Corrupt or unexpected file format: start fresh.
            self.memories = []

    def _save(self) -> None:
        payload = [asdict(memory) for memory in self.memories]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.memories = []
        self._save()

    def add_facts(self, facts: List[str], source_user_message: str) -> int:
        """Add deduplicated facts and persist to disk.

        Returns the count of newly stored facts.
        """
        existing = {memory.fact.strip().lower() for memory in self.memories}
        added = 0

        for fact in facts:
            cleaned = fact.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in existing:
                continue

            # Keep only the latest name memory to avoid conflicting identities.
            if "name is" in key and key.startswith("user"):
                self.memories = [m for m in self.memories if not m.fact.strip().lower().startswith("user's name is ")]
                existing = {memory.fact.strip().lower() for memory in self.memories}

            # Keep only the latest best-friend memory.
            if "best friend is" in key and key.startswith("user"):
                self.memories = [
                    m for m in self.memories if not m.fact.strip().lower().startswith("user's best friend is ")
                ]
                existing = {memory.fact.strip().lower() for memory in self.memories}

            self.memories.append(
                Memory(
                    fact=cleaned,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_user_message=source_user_message,
                )
            )
            existing.add(key)
            added += 1

        if added:
            self._save()
        return added

    def search(self, query: str, top_k: int = 5, min_score: float = 0.08) -> List[Memory]:
        query_vector = _term_freq(_tokenize(query))
        scored = []

        for memory in self.memories:
            score = _cosine_similarity(query_vector, _term_freq(_tokenize(memory.fact)))
            if score >= min_score:
                scored.append((score, memory.created_at, memory))

        # Prefer newer memories when similarity scores tie.
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [memory for _, _, memory in scored[:top_k]]
