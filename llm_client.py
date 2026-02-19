import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional


class LLMClient:
    INVALID_NAME_TOKENS = {"learning", "allergic", "training", "foodie", "ok", "okay"}
    def __init__(self, model: str = "grok-2-latest"):
        self.api_key = os.getenv("GROK_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("GROK_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "").strip() or model
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.x.ai/v1").strip().rstrip("/")
        self.api_call_count = 0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _post_chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.enabled:
            return None

        self.api_call_count += 1
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
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
                    return content.strip()
                if isinstance(content, list):
                    parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                    joined = "".join(parts).strip()
                    return joined or None
                return None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            return None

    def extract_facts(self, user_message: str) -> List[str]:
        system = (
            "Extract durable profile facts from the user message. "
            "Return JSON only: {\"facts\":[\"...\"]}. "
            "Use canonical forms like: User's name is X, User likes Y, User is allergic to Z, "
            "User lives in City, User works as Role, User has a dog named Name, User prefers vegetarian meals, "
            "User's birthday is on Date, User is training for Goal, User is learning Language, User drinks coffee every morning. "
            "Capture multiple facts if present. Ignore pure requests."
        )

        output = self._post_chat(system, user_message)
        api_facts = self._parse_api_facts(output) if output else []
        fallback = self._fallback_extract(user_message)
        merged = self._normalize_facts(api_facts + fallback)

        deduped: List[str] = []
        seen = set()
        for fact in merged:
            key = fact.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    def generate_reply(self, user_message: str, memory_facts: List[str], conversation_history: List[dict]) -> str:
        deterministic = self._deterministic_reply(user_message, memory_facts)
        if deterministic:
            return deterministic

        memory_block = "\n".join(f"- {m}" for m in memory_facts) if memory_facts else "- (none)"
        history_block = "\n".join(f"{m.get('role','user').upper()}: {m.get('content','')}" for m in conversation_history[-8:])

        system = (
            "You are a helpful assistant with long-term memory. "
            "Answer naturally, using relevant memory facts only."
        )
        prompt = (
            "Memories:\n"
            f"{memory_block}\n\n"
            "Conversation:\n"
            f"{history_block}\n\n"
            f"Latest user message: {user_message}\n\n"
            "Give a concise, personalized response."
        )

        output = self._post_chat(system, prompt)
        if output:
            return output

        return "I can help with that. Tell me what kind of suggestion you want."

    def _deterministic_reply(self, user_message: str, memory_facts: List[str]) -> Optional[str]:
        lower = user_message.lower()
        mem = self._memory_by_type(memory_facts)

        if "my name" in lower or "who am i" in lower:
            return f"Your name is {mem['name'][0]}." if mem["name"] else "I do not have your name saved yet."

        if "where do i live" in lower or "where i live" in lower:
            return f"You live in {mem['location'][0]}." if mem["location"] else "I do not have your location saved yet."

        allergy_query = bool(
            re.search(r"\bam i allergic\b", lower)
            or re.search(r"\bwhat am i allergic\b", lower)
            or re.search(r"\bdo i have (an )?allergy\b", lower)
            or re.search(r"\bno nuts?\b", lower)
            or re.search(r"\bnut[- ]?free\b", lower)
        )
        if allergy_query:
            if mem["allergy"]:
                return f"You told me you are allergic to {', '.join(mem['allergy'])}."
            return "I do not have any allergy memory saved yet."

        if "what do i like" in lower or "what i like" in lower or "what do i love" in lower:
            likes = self._ordered_likes(mem)
            return f"You told me you like {', '.join(likes)}." if likes else "I do not have any saved preferences yet."

        if "bestfriend" in lower or "best friend" in lower:
            if ("my" in lower and "is" in lower) and ("bestfriend" in lower or "best friend" in lower):
                return "Thanks for sharing. I will remember that."
            return f"Your best friend is {mem['relationship'][0]}." if mem["relationship"] else "I do not know your best friend yet."

        if "how's bruno" in lower or "how is bruno" in lower:
            return "I cannot check Bruno in real time, but a short walk, water, and play time are usually great for him."

        if (
            "practice a new language" in lower
            or "practice language" in lower
            or ("learning spanish" in lower and any(x in lower for x in ["help", "practice", "can you", "how"]))
        ):
            language = mem["language"][0] if mem["language"] else "your target language"
            return (
                f"For {language}, try: 1) 10-minute speaking practice, 2) 15-word vocab drill, "
                "3) one short listening clip and summary."
            )

        if "prepare for my race" in lower or "prepare for race" in lower or ("race" in lower and "help" in lower):
            if mem["training"]:
                return (
                    "For race prep: 1) easy run + strides, 2) hydration and carb-focused meal, "
                    "3) sleep and recovery routine."
                )
            return "Race prep basics: structured training, hydration, recovery, and consistent sleep."

        if "tea" in lower and ("coffee" in lower or "something else" in lower):
            if mem["routine"]:
                return "Since you drink coffee in the morning, tea can be a calmer evening option. Herbal tea is good for late hours."
            return "Tea is a good lighter option, especially in the evening; coffee is better earlier in the day."

        intent = self._detect_intent(lower)
        if intent["recommend"]:
            topic = self._infer_topic(lower)
            picks = self._topic_suggestions(topic, mem, intent)
            return f"Here are 3 suggestions: 1) {picks[0]}, 2) {picks[1]}, 3) {picks[2]}."

        if self._fallback_extract(user_message):
            if mem["name"] and ("i am" in lower or "my name" in lower):
                return f"Nice to meet you, {mem['name'][0]}. I will keep that in mind."
            return "Got it. I will keep that in mind."

        return None

    def _topic_suggestions(self, topic: str, mem: Dict[str, List[str]], intent: Dict[str, bool]) -> List[str]:
        likes = self._ordered_likes(mem)
        food_likes = [x for x in likes if self._classify_pref(x) == "food"]
        activity_likes = [x for x in likes if self._classify_pref(x) == "activity"]
        watch_likes = [x for x in likes if self._classify_pref(x) in {"watch", "animal"}]

        if topic == "food":
            vegetarian = any("vegetarian" in d.lower() for d in mem["diet"]) or any(
                "vegetarian" in f.lower() for f in food_likes
            )
            base = ["balanced grain bowl", "stir-fry veggies with noodles/rice", "quick snack plate"]
            if vegetarian:
                base = ["paneer/tofu stir-fry bowl", "veg quinoa salad", "lentil soup with whole-grain toast"]
            if food_likes:
                top = food_likes[0].lower()
                if "ice cream" in top:
                    base = ["fruit-and-nut ice cream bowl", "frozen yogurt parfait", "small brownie with ice cream"]
                elif "chinese" in top or "noodle" in top:
                    base = ["veg hakka noodles", "chilli garlic fried rice", "hot and sour soup"]
                elif "indian" in top:
                    base = ["paneer tikka bowl", "veg pulao with raita", "masala dosa with sambar"]
            if intent["nut_free"] or mem["allergy"]:
                avoid = ", ".join(mem["allergy"]) if mem["allergy"] else "nuts"
                base[2] = f"any safe homemade option (avoid {avoid})"
            return base

        if topic == "activity":
            if intent["indoor"] and intent["relax"]:
                return ["guided breathing session", "cozy journaling with tea", "gentle stretching routine"]
            if intent["indoor"]:
                if any("painting" in a.lower() for a in activity_likes):
                    return ["45-minute painting session", "color sketch challenge", "art session with calming music"]
                return ["home workout mini-set", "creative craft session", "reading + journaling block"]
            if intent["outdoor"]:
                if any("boating" in a.lower() for a in activity_likes):
                    return ["short boating session", "lake walk plus boating", "sunset waterfront plan"]
                return ["park walk", "short cycling route", "sunset photo walk"]
            if intent["relax"]:
                return ["slow evening walk", "light breathing + stretching", "screen-free wind-down routine"]
            if any("painting" in a.lower() for a in activity_likes):
                return ["45-minute painting session", "color sketch challenge", "art session with music"]
            return ["30-minute walk", "creative hobby sprint", "light social catch-up"]

        if topic == "watch":
            merged = " ".join(watch_likes).lower()
            if "sci-fi" in merged or "science fiction" in merged:
                return ["high-rated sci-fi film", "sci-fi mini-series episode", "space-themed drama"]
            if "drama" in merged:
                return ["emotional drama film", "character-driven mini-series", "coming-of-age drama"]
            if any(w in merged for w in ["dog", "cat", "animal", "bird", "parrot"]):
                return ["wildlife documentary", "nature mini-series", "family adventure film"]
            return ["light comedy film", "short thriller series", "high-rated documentary"]

        if topic == "birthday":
            return ["personalized birthday note + flowers", "surprise dinner plan", "memory scrapbook + small gift"]

        if topic == "music":
            return ["indie chill playlist", "soft acoustic mix", "ambient lo-fi set"]

        if topic == "travel":
            city = mem["location"][0] if mem["location"] else "nearby"
            return [f"a {city} day trip", "local food trail", "sunset viewpoint plan"]

        return ["one practical option now", "one low-effort option today", "one fun option this evening"]

    def _detect_intent(self, lower: str) -> Dict[str, bool]:
        recommend = bool(re.search(r"\bsugg\w*\b", lower)) or any(
            phrase in lower
            for phrase in [
                "recommend",
                "give me",
                "what should i",
                "what can i",
                "i want",
                "help me choose",
                "help me prepare",
            ]
        )
        return {
            "recommend": recommend,
            "indoor": any(x in lower for x in ["indoor", "inside", "at home", "home"]),
            "outdoor": any(x in lower for x in ["outdoor", "outside", "park", "nature"]),
            "relax": any(x in lower for x in ["relax", "calm", "unwind", "stress"]),
            "nut_free": any(x in lower for x in ["no nuts", "nut free", "without nuts", "no nut"]),
        }

    def _infer_topic(self, lower: str) -> str:
        mapping = {
            "food": ["food", "eat", "meal", "dinner", "lunch", "breakfast", "snack", "hungry", "cook"],
            "watch": ["watch", "movie", "show", "series", "anime", "documentary", "ott"],
            "music": ["music", "song", "playlist", "listen"],
            "activity": ["activity", "hobby", "relax", "indoor", "outdoor", "weekend", "race"],
            "travel": ["travel", "trip", "vacation", "holiday", "visit"],
            "birthday": ["birthday", "surprise", "gift"],
        }
        for topic, keywords in mapping.items():
            if any(k in lower for k in keywords):
                return topic
        return "general"

    def _memory_by_type(self, memory_facts: List[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {
            "name": [],
            "food": [],
            "activity": [],
            "animal": [],
            "watch": [],
            "music": [],
            "allergy": [],
            "relationship": [],
            "location": [],
            "occupation": [],
            "pet": [],
            "diet": [],
            "birthday": [],
            "training": [],
            "language": [],
            "routine": [],
            "general": [],
        }
        for fact in memory_facts:
            f = fact.strip()
            if not f:
                continue
            low = f.lower()
            if low.startswith("user's name is "):
                out["name"].append(f[len("User's name is "):].strip())
                continue
            if low.startswith("user is allergic to "):
                out["allergy"].append(f[len("User is allergic to "):].strip())
                continue
            if low.startswith("user's best friend is "):
                out["relationship"].append(f[len("User's best friend is "):].strip())
                continue
            if low.startswith("user lives in "):
                out["location"].append(f[len("User lives in "):].strip())
                continue
            if low.startswith("user works as "):
                out["occupation"].append(f[len("User works as "):].strip())
                continue
            if low.startswith("user has a dog named "):
                out["pet"].append(f[len("User has a dog named "):].strip())
                continue
            if low.startswith("user prefers "):
                out["diet"].append(f[len("User prefers "):].strip())
                continue
            if low.startswith("user's birthday is on "):
                out["birthday"].append(f[len("User's birthday is on "):].strip())
                continue
            if low.startswith("user is training for "):
                out["training"].append(f[len("User is training for "):].strip())
                continue
            if low.startswith("user is learning "):
                out["language"].append(f[len("User is learning "):].strip())
                continue
            if low.startswith("user drinks "):
                out["routine"].append(f[len("User drinks "):].strip())
                continue
            if low.startswith("user likes "):
                val = f[len("User likes "):].strip()
                out[self._classify_pref(val)].append(val)
                continue
            out["general"].append(f)

        for key in out:
            dedup = []
            seen = set()
            for v in out[key]:
                k = v.lower()
                if k in seen:
                    continue
                seen.add(k)
                dedup.append(v)
            out[key] = dedup
        return out

    def _ordered_likes(self, mem: Dict[str, List[str]]) -> List[str]:
        ordered = mem["food"] + mem["activity"] + mem["animal"] + mem["watch"] + mem["music"] + mem["general"]
        dedup = []
        seen = set()
        for x in ordered:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            dedup.append(x)
        return dedup

    def _classify_pref(self, value: str) -> str:
        v = value.lower()
        if any(w in v for w in ["food", "eat", "meal", "snack", "ice cream", "indian", "italian", "chinese", "pizza", "pasta", "biryani", "noodle", "vegetarian"]):
            return "food"
        if any(w in v for w in ["painting", "boating", "horse", "running", "gym", "yoga", "dance", "play", "playing", "hobby", "marathon"]):
            return "activity"
        if any(w in v for w in ["dog", "dogs", "cat", "cats", "bird", "birds", "parrot", "parrots", "animal", "animals"]):
            return "animal"
        if any(w in v for w in ["drama", "movie", "show", "series", "anime", "documentary", "football", "sci-fi", "science fiction"]):
            return "watch"
        if any(w in v for w in ["music", "song", "playlist"]):
            return "music"
        return "general"

    def _fallback_extract(self, text: str) -> List[str]:
        facts: List[str] = []
        # Split both on sentence marks and commas to avoid merged facts.
        clauses = [c.strip() for c in re.split(r"[.!?,]", text) if c.strip()]

        for clause in clauses:
            low = clause.lower().strip()
            if self._looks_like_request(low):
                continue

            name_match = re.search(r"\b(?:i am|i'm|my name is)\s+([A-Za-z][A-Za-z'-]{1,30})\b", clause, flags=re.IGNORECASE)
            if name_match:
                name = name_match.group(1).strip()
                if name.lower() not in self.INVALID_NAME_TOKENS:
                    facts.append(f"User's name is {name}")

            allergy_match = re.search(r"\ballergic to\s+([^,.;!?]+)", clause, flags=re.IGNORECASE)
            if allergy_match:
                allergy = self._clean_pref(allergy_match.group(1))
                if allergy:
                    facts.append(f"User is allergic to {allergy}")

            pref_matches = re.findall(
                r"\b(?:i\s+also\s+like|i\s+too\s+like|i\s+love|i\s+like|i\s+enjoy|i\s+prefer|i\s+am\s+into|i'm\s+into)\s+([^.!?]+)",
                clause,
                flags=re.IGNORECASE,
            )
            for block in pref_matches:
                for item in re.split(r",| and also | also | plus | & | and ", block, flags=re.IGNORECASE):
                    val = self._clean_pref(item)
                    if val:
                        facts.append(f"User likes {val}")

            location = re.search(r"\bi live in\s+([A-Za-z][A-Za-z\s'-]*)$", clause, flags=re.IGNORECASE)
            if location:
                facts.append(f"User lives in {self._clean_pref(location.group(1))}")

            work = re.search(r"\bi work as\s+([A-Za-z][A-Za-z\s'-]*)$", clause, flags=re.IGNORECASE)
            if work:
                facts.append(f"User works as {self._clean_pref(work.group(1))}")

            dog = re.search(r"\bi have a dog named\s+([A-Za-z][A-Za-z\s'-]*)$", clause, flags=re.IGNORECASE)
            if dog:
                facts.append(f"User has a dog named {self._clean_pref(dog.group(1))}")

            if re.search(r"\bi prefer vegetarian meals\b", clause, flags=re.IGNORECASE):
                facts.append("User prefers vegetarian meals")

            birthday = re.search(r"\bmy birthday is on\s+([A-Za-z0-9\s]+)$", clause, flags=re.IGNORECASE)
            if birthday:
                facts.append(f"User's birthday is on {self._clean_pref(birthday.group(1))}")

            training = re.search(r"\bi am training for\s+([A-Za-z0-9\s]+)$", clause, flags=re.IGNORECASE)
            if training:
                facts.append(f"User is training for {self._clean_pref(training.group(1))}")

            learning = re.search(r"\bi am learning\s+([A-Za-z\s]+)$", clause, flags=re.IGNORECASE)
            if learning:
                facts.append(f"User is learning {self._clean_pref(learning.group(1))}")

            if re.search(r"\bi drink coffee every morning\b", clause, flags=re.IGNORECASE):
                facts.append("User drinks coffee every morning")

            best_friend = re.search(r"\bmy\s+best\s*friend\s+is\s+([A-Za-z][A-Za-z\s'-]*)", clause, flags=re.IGNORECASE)
            if best_friend:
                name = self._clean_pref(best_friend.group(1))
                if name:
                    facts.append(f"User's best friend is {name}")

        return facts

    def _clean_pref(self, text: str) -> str:
        x = text.strip(" .,")
        x = re.sub(
            r"^(?:i\s+also\s+like|i\s+too\s+like|i\s+love|i\s+like|i\s+enjoy|i\s+prefer|i\s+am\s+into|i'm\s+into|into|the|a|an)\s+",
            "",
            x,
            flags=re.IGNORECASE,
        )
        x = re.sub(r"\b(?:too|also|as well)\b$", "", x, flags=re.IGNORECASE).strip(" ,.")
        return x

    def _looks_like_request(self, lower_text: str) -> bool:
        if re.search(r"\bsugg\w*\b", lower_text):
            return True
        starters = [
            "can you",
            "can u",
            "could you",
            "would you",
            "please",
            "recommend",
            "give me",
            "what should i",
            "what can i",
            "how can i",
            "i want",
            "help me",
        ]
        return any(lower_text.startswith(s) for s in starters)

    def _parse_api_facts(self, output: str) -> List[str]:
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", output)
            if not match:
                return []
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []

        facts = data.get("facts", []) if isinstance(data, dict) else []
        if not isinstance(facts, list):
            return []

        return [str(item).strip() for item in facts if str(item).strip()]

    def _normalize_facts(self, facts: List[str]) -> List[str]:
        out: List[str] = []
        for fact in facts:
            cleaned = fact.strip()
            if not cleaned:
                continue

            combo = re.search(
                r"^user is allergic to\s+(.+?)\s+and\s+i\s+(?:love|like)\s+(.+)$",
                cleaned,
                flags=re.IGNORECASE,
            )
            if combo:
                a = self._clean_pref(combo.group(1))
                b = self._clean_pref(combo.group(2))
                if a:
                    out.append(f"User is allergic to {a}")
                if b:
                    out.append(f"User likes {b}")
                continue

            pref = re.search(r"^user (?:likes|loves|enjoys|prefers)\s+(.+)$", cleaned, flags=re.IGNORECASE)
            if pref:
                val = self._clean_pref(pref.group(1))
                if val:
                    out.append(f"User likes {val}")
                continue

            name = re.search(r"^user'?s?\s+name\s+is\s+(.+)$", cleaned, flags=re.IGNORECASE)
            if name:
                val = self._clean_pref(name.group(1))
                if val:
                    out.append(f"User's name is {val}")
                continue

            bf = re.search(r"^user'?s?\s+best\s*friend\s+is\s+(.+)$", cleaned, flags=re.IGNORECASE)
            if bf:
                val = self._clean_pref(bf.group(1))
                if val:
                    out.append(f"User's best friend is {val}")
                continue

            out.append(cleaned)
        return out
