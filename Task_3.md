# **Task 3 — How Should AI Decide What to Remember?**

If an AI assistant has limited memory capacity, it cannot store everything a user says. So the real challenge is not storage — it is prioritization. The AI must intelligently decide what is important, what is temporary, and what can be safely forgotten.

In my opinion, memory selection should not rely on just one strategy. It should combine multiple signals like importance, frequency, recency, and user feedback.

---

## **Approach 1: Importance Scoring Using Keywords \+ Text Processing**

One practical approach is to score memories based on how important the content appears.

Before storing anything, the system can apply **text preprocessing**:

* Remove stop words (like “is”, “the”, “and”) — this alone can remove around 30% of low-value data.

* Normalize words (lowercasing, stemming).

* Extract meaningful keywords.

Then, we can assign higher weights to important keywords such as:

* allergy

* birthday

* deadline

* exam

* address

* medical

* preference

This is similar to keyword-weighting models (like TF-IDF style scoring). But instead of just counting words, we can also use **encoding (embeddings)** to understand the meaning of the sentence. This allows the AI to recognize similar meanings even if the wording changes.

For example:

* “I am allergic to peanuts”

* “Peanuts cause me a reaction”

Even though the words differ, encoding helps prioritize both.

### **Trade-offs**

Strengths:

* Efficient and structured.

* Reduces storage of unnecessary conversational filler.

* More scalable than storing raw text.

Weaknesses:

* Might miss emotional or contextual importance.

* Keyword weighting alone can fail if the wording is unusual.

So this approach can work best when combined with semantic encoding.

---

## 

## 

## **Approach 2: Recency and Frequency-Based Retention**

Another intelligent strategy is combining:

1. **Recency-based decay**  
    Recent information gets higher priority. If something hasn’t been mentioned in a long time, its score gradually decreases.

2. **Frequency-based reinforcement**  
    If a user repeatedly mentions something, it becomes more important and needed.

This mimics human memory:

* We remember what we repeat.

* We forget what we don’t use.

For example:

* If a user mentions exam preparation multiple times, that should stay.

* If they mention a random movie once and never again, it can fade.

### **Trade-offs**

Strengths:

* Naturally adaptive.

* Simple to implement.

* Reflects real conversational patterns.

Weaknesses:

* Rare but critical information might be forgotten.

* Frequency does not always equal importance.

That’s why frequency and recency should not be the only signals.

---

## **Approach 3: User-Controlled Memory (Freedom of Selection)**

One thing I strongly believe should be included is **user freedom**.

The AI can allow users to:

* Mark a memory as important.

* Pin certain responses.

* Delete stored memories.

* Approve what gets saved.

Just like how ChatGPT asks us : Which response do we prefer? Helps to train itself better.

This solves a major problem: the AI does not always know what is personally important. Giving the user control increases transparency and trust.

Instead of blindly deciding, the assistant becomes collaborative.

### **Trade-offs**

**Strengths:**

* High user trust and transparency.

* Reduces incorrect memory retention.

* Empowers personalization.

**Weaknesses:**

* Requires extra user effort.

* Some users may ignore manual selection.

---

## **My Opinion: What Works Best?**

I think the best solution is a hybrid system:

1. Preprocess text to remove unnecessary noise.

2. Use keyword importance scoring.

3. Apply encoding to understand semantic meaning.

4. Add recency and frequency weighting.

5. Allow user-controlled memory overrides.

 A purely keyword-based system is too rigid.  
 A purely recency-based system forgets long-term facts.  
 A purely frequency-based system may overvalue repetition.

But combining structured scoring, semantic understanding, and user control creates a balanced system.

Overall, an AI assistant should behave less like a storage database and more like an intelligent filter remembering what truly matters, forgetting what doesn’t, and giving users the final say when necessary.

This is my take on the question 

*“If an AI assistant has limited memory capacity, how should it decide what information to remember and what*  
*to forget? What strategies or algorithms could help it make this decision intelligently?”.*

***Thank you for reviewing the report.***