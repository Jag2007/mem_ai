
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
PREF_PATTERN = re.compile(r"^User (likes|dislikes)\s+(.+)$", flags=re.IGNORECASE)
INVALID_NAME_TOKENS = {"learning", "allergic", "training", "foodie", "ok", "okay", "preparing"}


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
            loaded = [Memory(**item) for item in raw]
            # Clean legacy markers from older versions.
            for m in loaded:
                m.fact = m.fact.replace(" (needs confirmation)", "").strip()
            self.memories = [m for m in loaded if not self._is_invalid_memory(m.fact)]
            self._compact_preference_memories()
            self._compact_single_value_memories()
            if len(self.memories) != len(loaded):
                self._save()
        except (json.JSONDecodeError, TypeError, KeyError):
            # Corrupt or unexpected file format: start fresh.
            self.memories = []

    def _save(self) -> None:
        payload = [asdict(memory) for memory in self.memories]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _is_invalid_memory(self, fact: str) -> bool:
        f = fact.strip().lower()
        # Drop low-information fragments accidentally extracted from questions.
        bad_exact = {"user likes to watch", "user likes to eat", "user likes to play", "user likes watch"}
        if f in bad_exact:
            return True
        if f.startswith("user likes to ") and len(f.split()) <= 4:
            return True
        return False

    def clear(self) -> None:
        self.memories = []
        self._save()

    def _normalize_pref_item(self, text: str) -> str:
        cleaned = text.strip().lower()
        cleaned = re.sub(r"[^a-z0-9\s-]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _is_near_pref_match(self, a: str, b: str) -> bool:
        if a == b:
            return True
        if abs(len(a) - len(b)) > 1:
            return False
        if (a in b or b in a) and min(len(a), len(b)) >= 4:
            return True
        # one-edit-distance check
        i = 0
        j = 0
        edits = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
                j += 1
                continue
            edits += 1
            if edits > 1:
                return False
            if len(a) > len(b):
                i += 1
            elif len(b) > len(a):
                j += 1
            else:
                i += 1
                j += 1
        if i < len(a) or j < len(b):
            edits += 1
        return edits <= 1

    def _upsert_preference_fact(self, fact: str, source_user_message: str) -> int:
        match = PREF_PATTERN.match(fact.strip())
        if not match:
            return 0
        polarity = match.group(1).lower()  # likes | dislikes
        item = match.group(2).strip()
        if not item:
            return 0
        item_key = self._normalize_pref_item(item)
        if not item_key:
            return 0

        # Find any existing like/dislike entries for the same (or near-same) item.
        matched_indexes = []
        canonical_item = item
        for idx, memory in enumerate(self.memories):
            m = PREF_PATTERN.match(memory.fact.strip())
            if not m:
                continue
            existing_item = m.group(2).strip()
            existing_key = self._normalize_pref_item(existing_item)
            if existing_key and self._is_near_pref_match(item_key, existing_key):
                matched_indexes.append(idx)
                # Prefer the longer token as canonical label (pizza over pizz).
                if len(existing_item) > len(canonical_item):
                    canonical_item = existing_item

        # If current input is longer, keep it as canonical display.
        if len(item) >= len(canonical_item):
            canonical_item = item

        new_fact = f"User {polarity} {canonical_item}"
        now = datetime.now(timezone.utc).isoformat()

        if not matched_indexes:
            self.memories.append(
                Memory(
                    fact=new_fact,
                    created_at=now,
                    source_user_message=source_user_message,
                )
            )
            return 1

        # Replace first matched slot and remove remaining duplicates/opposites.
        first = matched_indexes[0]
        self.memories[first] = Memory(
            fact=new_fact,
            created_at=now,
            source_user_message=source_user_message,
        )
        for idx in reversed(matched_indexes[1:]):
            del self.memories[idx]
        return 0

    def _find_existing_pref_key(self, pref_keys: List[str], new_key: str) -> str:
        for key in pref_keys:
            if self._is_near_pref_match(key, new_key):
                return key
        return new_key

    def _compact_preference_memories(self) -> None:
        if not self.memories:
            return
        compacted: List[Memory] = []
        pref_latest: Dict[str, Memory] = {}

        for mem in self.memories:
            match = PREF_PATTERN.match(mem.fact.strip())
            if not match:
                compacted.append(mem)
                continue

            polarity = match.group(1).lower()
            item = match.group(2).strip()
            key = self._normalize_pref_item(item)
            if not key:
                continue
            resolved = self._find_existing_pref_key(list(pref_latest.keys()), key)

            existing = pref_latest.get(resolved)
            if existing:
                existing_match = PREF_PATTERN.match(existing.fact.strip())
                existing_item = item
                if existing_match:
                    existing_item = existing_match.group(2).strip()
                # Prefer longer canonical display label (pizza over pizz).
                canonical_item = item if len(item) >= len(existing_item) else existing_item
                pref_latest[resolved] = Memory(
                    fact=f"User {polarity} {canonical_item}",
                    created_at=mem.created_at,
                    source_user_message=mem.source_user_message,
                )
            else:
                pref_latest[resolved] = Memory(
                    fact=f"User {polarity} {item}",
                    created_at=mem.created_at,
                    source_user_message=mem.source_user_message,
                )

        compacted.extend(pref_latest.values())
        # Keep stable ordering by creation time.
        compacted.sort(key=lambda m: m.created_at)
        self.memories = compacted

    def _single_value_group(self, fact: str) -> str:
        low = fact.strip().lower()
        if low.startswith("user's name is "):
            return "name"
        if low.startswith("user's birthday is on "):
            return "birthday"
        if low.startswith("user lives in "):
            return "location"
        if low.startswith("user works as ") or low.startswith("user studies "):
            return "occupation"
        if low.startswith("user's best friend is "):
            return "best_friend"
        if low.startswith("user has a dog named "):
            return "dog_name"
        return ""

    def _compact_single_value_memories(self) -> None:
        if not self.memories:
            return
        latest: Dict[str, Memory] = {}
        others: List[Memory] = []
        for mem in self.memories:
            group = self._single_value_group(mem.fact)
            if not group:
                others.append(mem)
                continue
            if group == "name":
                low = mem.fact.strip().lower()
                value = low.replace("user's name is", "", 1).strip()
                if value in INVALID_NAME_TOKENS:
                    continue
            latest[group] = mem
        merged = others + list(latest.values())
        merged.sort(key=lambda m: m.created_at)
        self.memories = merged

    def list_facts(self) -> List[str]:
        return [m.fact for m in self.memories]

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
            # Preferences are upserted in-place (latest statement wins).
            if PREF_PATTERN.match(cleaned):
                added += self._upsert_preference_fact(cleaned, source_user_message)
                continue
            group = self._single_value_group(cleaned)
            if group:
                self.memories = [m for m in self.memories if self._single_value_group(m.fact) != group]
                self.memories.append(
                    Memory(
                        fact=cleaned,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        source_user_message=source_user_message,
                    )
                )
                added += 1
                existing = {memory.fact.strip().lower() for memory in self.memories}
                continue
            key = cleaned.lower()
            if key in existing:
                continue

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
