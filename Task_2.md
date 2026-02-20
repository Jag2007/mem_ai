# **Task 2 — Memory Quality Evaluator**

## **1\. Introduction**

This document evaluates the performance of the memory retrieval system built in Task 1\.  
 The goal of this evaluation is to measure how accurately the system retrieves the correct stored memory when given a related query.

The evaluation focuses on factual recall accuracy and relevance quality.

---

# **2\. Test Dataset**

## **2.1 Stored Memories**

The following 10 factual memories were stored in the memory system:

1. My name is Priya Sharma.

2. I live in Bangalore.

3. I am allergic to peanuts,almonds.

4. My favorite food is pasta.

5. I work as an AI engineer.

6. My birthday is on April 12\.

7. I have a cat named Luna.

8. I love watching sci-fi movies.

9. I dislike apples.

10. I am currently learning machine learning.

Each memory contains a single clear fact to reduce ambiguity and improve evaluation accuracy.

---

## **2.2 Test Queries**

The following 10 queries were used to test retrieval performance:

1. What is my name?

2. Where do I live?

3. Do I have any allergies?

4. What is my favorite food?

5. What do I do for work?

6. When is my birthday?

7. Do I have a cat ?

8. What kind of movies do I like?

9. What do i dislike?

10. What am I currently studying?

Each query directly corresponds to one stored memory.

---

# **3\. Evaluation Methodology**

To measure the quality of the memory system, the following metrics were used:

## **Accuracy**

Accuracy measures whether the system retrieved the exact correct memory.

Accuracy \= (Number of correct retrievals / Total queries) × 100

This metric evaluates strict correctness.

---

# **4\. Evaluation Results**

## **4.1 Query-Level Evaluation Table**

| Query No. | Query                          | Answered Correctly (Yes / No) |
| --------- | ------------------------------ | ----------------------------- |
| 1         | What is my name?               | Yes                           |
| 2         | Where do I live?               | Yes                           |
| 3         | Do I have any allergies?       | Yes                           |
| 4         | What is my favorite food?      | Yes                           |
| 5         | What do I do for work?         | Yes                           |
| 6         | When is my birthday?           | Yes                           |
| 7         | Do I have a cat?               | Yes                           |
| 8         | What kind of movies do I like? | Yes                           |
| 9         | What do i dislike?             | Yes                           |
| 10        | What am I currently studying?  | No                            |

---

## **4.2 Summary Results**

Total Queries: 10  
 Correct Retrievals: 9
Accuracy: 90 %

---

# **5\. Observations**

## **What Worked**

- The system performed well for direct factual questions.

- Queries with similar wording to stored memories showed high accuracy.

- Clear one-fact-per-memory structure improved retrieval precision.

## **What Failed**

- Slight variations in phrasing reduced exact match accuracy.

- Semantic variations (e.g., "profession" vs "work") and similar words with a differing verb form also impacted performance .

- Some responses were partially relevant but not exact matches.

---

# **6\. Conclusion**

The memory system demonstrates reliable performance for structured factual recall. Accuracy is high when query wording closely matches stored memory phrasing.

However, performance decreases when semantic variation increases. This indicates that the current retrieval approach may rely heavily on keyword matching.

To improve performance, future enhancements may include:

- Embedding-based semantic similarity

- Context-aware ranking

- Improved scoring mechanisms

Overall, the system performs well for direct keyword-aligned queries but lacks robustness when handling paraphrased or semantically similar inputs. This highlights the need for more advanced semantic retrieval techniques to make the system production-ready and adaptable to real-world query variations.
