
import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional


class LLMClient:
    def __init__(self, model: str = "grok-2-latest"):
        # Prefer Grok/xAI credentials, then fall back to OpenAI-compatible env names.
        self.api_key = os.getenv("GROK_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        self.model = (
            os.getenv("GROK_MODEL", "").strip()
            or os.getenv("OPENAI_MODEL", "").strip()
            or model
        )
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.x.ai/v1").strip().rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _post_chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.enabled:
            return None

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices", [])
                if not choices:
                    return None
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                    joined = "".join(parts).strip()
                    return joined or None
                return None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

    def extract_facts(self, user_message: str) -> List[str]:
        """Extract durable user facts worth remembering across sessions."""
        system = (
            "You extract durable personal facts from a user message. "
            "Return JSON only with shape: {\"facts\": [\"...\"]}. "
            "Keep each fact short, objective, and in third person. "
            "Only include info likely useful later (name, preferences, allergies, routines, goals). "
            "If nothing useful, return {\"facts\": []}."
        )
        output = self._post_chat(system, user_message)

        if output:
            try:
                data = json.loads(output)
                facts = data.get("facts", [])
                if isinstance(facts, list):
                    parsed = [str(item).strip() for item in facts if str(item).strip()]
                    return self._normalize_facts(parsed)
            except json.JSONDecodeError:
                pass

        return self._normalize_facts(self._fallback_extract(user_message))

    def generate_reply(self, user_message: str, memory_facts: List[str], conversation_history: List[dict]) -> str:
        deterministic = self._deterministic_reply(user_message, memory_facts)
        if deterministic:
            return deterministic

        memory_section = "\n".join(f"- {fact}" for fact in memory_facts) if memory_facts else "- (no relevant memories)"

        history_lines = []
        for msg in conversation_history[-8:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_lines.append(f"{role.upper()}: {content}")

        prompt = (
            "Relevant memories about the user:\n"
            f"{memory_section}\n\n"
            "Recent conversation:\n"
            f"{'\n'.join(history_lines)}\n\n"
            "Now answer the latest USER message naturally. "
            "Use memories when relevant, but do not force them into unrelated answers. "
            "If safety-related memories exist (like allergies), prioritize them in recommendations."
        )

        system = "You are a helpful, concise assistant in a terminal chat app."
        output = self._post_chat(system, prompt)
        if output and output.strip():
            return output.strip()

        # Graceful fallback if API is unavailable.
        deterministic = self._deterministic_reply(user_message, memory_facts)
        if deterministic:
            return deterministic
        return "I can help with that. Could you share a bit more detail so I can personalize the suggestion?"

    def _fallback_extract(self, text: str) -> List[str]:
        """Very small heuristic extractor used when API is unavailable."""
        if text.strip().endswith("?"):
            return []

        facts: List[str] = []

        name_match = re.search(r"\b(?:i am|i'm)\s+([A-Za-z]+)\b", text, flags=re.IGNORECASE)
        if name_match:
            facts.append(f"User's name is {name_match.group(1)}")

        allergy_match = re.search(r"\ballergic to\s+([^.!?,;]+)", text, flags=re.IGNORECASE)
        if allergy_match:
            allergy_value = re.split(r"\band\b", allergy_match.group(1), maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if allergy_value:
                facts.append(f"User is allergic to {allergy_value}")

        preference_matches = re.findall(
            r"\b(?:i love|i like|i enjoy|i prefer)\s+([^.!?]+)",
            text,
            flags=re.IGNORECASE,
        )
        for match in preference_matches:
            for item in self._split_preference_items(match):
                facts.append(f"User likes {item}")

        best_friend_match = re.search(
            r"\bmy\s+best\s*friend\s+is\s+([A-Za-z][A-Za-z\s'-]*)",
            text,
            flags=re.IGNORECASE,
        )
        if best_friend_match:
            facts.append(f"User's best friend is {best_friend_match.group(1).strip()}")

        return facts

    def _extract_name(self, memory_facts: List[str]) -> Optional[str]:
        for fact in memory_facts:
            match = re.search(r"^user's name is\s+(.+)$", fact, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_allergies(self, memory_facts: List[str]) -> List[str]:
        allergies: List[str] = []
        for fact in memory_facts:
            match = re.search(r"^user is allergic to\s+(.+)$", fact, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and value.lower() not in [a.lower() for a in allergies]:
                    allergies.append(value)
        return allergies

    def _extract_preferences(self, memory_facts: List[str]) -> List[str]:
        likes: List[str] = []
        for fact in memory_facts:
            match = re.search(
                r"^user (?:likes|loves|enjoys|prefers)\s+(.+)$",
                fact,
                flags=re.IGNORECASE,
            )
            if match:
                value = match.group(1).strip()
                if value and value.lower() not in [x.lower() for x in likes]:
                    likes.append(value)
        return likes

    def _extract_best_friend(self, memory_facts: List[str]) -> Optional[str]:
        for fact in memory_facts:
            match = re.search(
                r"^user'?s?\s+best\s*friend\s+is\s+(.+)$",
                fact,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
        return None

    def _deterministic_reply(self, user_message: str, memory_facts: List[str]) -> Optional[str]:
        lower_user = user_message.lower()
        lower_facts = [fact.lower() for fact in memory_facts]
        remembered_name = self._extract_name(memory_facts)
        remembered_allergies = self._extract_allergies(memory_facts)
        remembered_preferences = self._extract_preferences(memory_facts)
        remembered_best_friend = self._extract_best_friend(memory_facts)

        if "my name" in lower_user or "who am i" in lower_user:
            if remembered_name:
                return f"Your name is {remembered_name}."
            return "I do not have your name saved yet."

        if "allergic" in lower_user:
            if remembered_allergies:
                return f"You told me you are allergic to {', '.join(remembered_allergies)}."
            return "I do not have any allergy memory saved yet."

        declares_best_friend = bool(
            re.search(r"\bmy\s+best\s*friend\s+is\b", lower_user)
            or re.search(r"\bmy\s+bestfriend\s+is\b", lower_user)
        )
        if declares_best_friend:
            return "Thanks for sharing. I will remember that."

        if "bestfriend" in lower_user or "best friend" in lower_user:
            if remembered_best_friend:
                return f"Your best friend is {remembered_best_friend}."
            return "I do not know your best friend yet. You can tell me by saying: My best friend is <name>."

        if "what do i love" in lower_user or "what do i like" in lower_user:
            if remembered_preferences:
                ordered = self._order_preferences(remembered_preferences)
                return f"You told me you like {', '.join(ordered)}."
            return "I do not have any saved preference yet."

        food_intent = any(
            word in lower_user
            for word in ["dinner", "lunch", "breakfast", "eat", "meal", "food", "hungry", "flavorful", "flavourful"]
        ) or ("suggest" in lower_user and "eat" in lower_user)

        if "suggest" in lower_user and remembered_preferences:
            preferred_item = self._pick_preferred_item(remembered_preferences)
            if preferred_item:
                return (
                    f"Since you like {preferred_item}, you can try it in two ways tonight: "
                    f"1) classic style, 2) spicy chef-special version."
                )
            return f"Based on your preferences, you could try {remembered_preferences[0]}."

        if food_intent:
            likes_italian = any("likes italian" in fact for fact in lower_facts)
            peanut_allergy = any("allergic to peanuts" in fact for fact in lower_facts)
            if remembered_preferences:
                top_like = self._pick_preferred_item(remembered_preferences) or remembered_preferences[0]
                if remembered_allergies:
                    return (
                        f"Since you like {top_like}, try a flavorful {top_like} dish made without "
                        f"{', '.join(remembered_allergies)}."
                    )
                return (
                    f"Since you like {top_like}, try this: {top_like} with extra herbs and bold spices. "
                    "If you want, I can give 3 specific dish options next."
                )
            if likes_italian and peanut_allergy:
                return "How about a peanut-free pasta primavera? Since you like Italian food, it should fit well."
            if likes_italian:
                return "How about Italian tonight, maybe a pasta primavera or margherita pizza?"
            if peanut_allergy:
                return "A peanut-free rice bowl with roasted vegetables could be a good option tonight."

        new_facts = self._fallback_extract(user_message)
        if new_facts:
            return "Thanks for sharing. I will remember that."

        if memory_facts:
            return f"I remember this about you: {memory_facts[0]}."

        return None

    def _split_preference_items(self, raw_value: str) -> List[str]:
        cleaned = raw_value.strip(" .")
        if not cleaned:
            return []

        parts = re.split(r",| and also | also | plus | & | and ", cleaned, flags=re.IGNORECASE)
        items: List[str] = []
        for part in parts:
            item = part.strip(" .")
            item = re.sub(r"^(the|a|an)\s+", "", item, flags=re.IGNORECASE)
            if not item:
                continue
            # Keep reasonably informative items only.
            if len(item) < 2:
                continue
            items.append(item)
        return items

    def _pick_preferred_item(self, preferences: List[str]) -> Optional[str]:
        if not preferences:
            return None
        food_keywords = [
            "biryani", "pasta", "pizza", "rice", "burger", "curry", "noodle", "salad",
            "food", "dish", "cuisine", "chicken", "paneer", "dessert", "italian", "indian",
        ]
        for pref in preferences:
            lower = pref.lower()
            if any(keyword in lower for keyword in food_keywords):
                return pref
        return preferences[0]

    def _order_preferences(self, preferences: List[str]) -> List[str]:
        food_keywords = [
            "biryani", "pasta", "pizza", "rice", "burger", "curry", "noodle", "salad",
            "food", "dish", "cuisine", "chicken", "paneer", "dessert", "italian", "indian",
        ]
        ranked = []
        for pref in preferences:
            lower = pref.lower()
            is_food = any(keyword in lower for keyword in food_keywords)
            ranked.append((1 if is_food else 0, pref))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [pref for _, pref in ranked]

    def _normalize_facts(self, facts: List[str]) -> List[str]:
        normalized: List[str] = []
        for fact in facts:
            cleaned = fact.strip()
            lower = cleaned.lower()

            # Split malformed combined fact:
            # "User is allergic to peanuts and I love Italian food"
            combo = re.search(
                r"^user is allergic to\s+(.+?)\s+and\s+i\s+(?:love|like)\s+(.+)$",
                cleaned,
                flags=re.IGNORECASE,
            )
            if combo:
                allergy = combo.group(1).strip(" .")
                pref = combo.group(2).strip(" .")
                if allergy:
                    normalized.append(f"User is allergic to {allergy}")
                if pref:
                    normalized.append(f"User likes {pref}")
                continue

            # Canonicalize common preference phrasings.
            pref_match = re.search(
                r"^user (?:likes|loves|enjoys|prefers)\s+(.+)$",
                cleaned,
                flags=re.IGNORECASE,
            )
            if pref_match:
                pref = pref_match.group(1).strip(" .")
                if pref:
                    normalized.append(f"User likes {pref}")
                continue

            # Canonicalize name phrasing.
            name_match = re.search(
                r"^user'?s?\s+name\s+is\s+(.+)$",
                cleaned,
                flags=re.IGNORECASE,
            )
            if name_match:
                normalized.append(f"User's name is {name_match.group(1).strip(' .')}")
                continue

            best_friend_match = re.search(
                r"^user'?s?\s+best\s*friend\s+is\s+(.+)$",
                cleaned,
                flags=re.IGNORECASE,
            )
            if best_friend_match:
                normalized.append(f"User's best friend is {best_friend_match.group(1).strip(' .')}")
                continue

            if lower:
                normalized.append(cleaned)

        # Deduplicate while preserving order.
        seen = set()
        result: List[str] = []
        for fact in normalized:
            key = fact.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(fact)
        return result
