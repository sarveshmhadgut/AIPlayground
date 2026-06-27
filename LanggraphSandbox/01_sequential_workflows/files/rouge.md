ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used for evaluating the quality of automatically generated text, primarily in summarization and machine translation. It works by comparing an automatically produced summary (candidate) against one or more human-written summaries (references) by counting the overlap of n-grams or subsequences.

The most common variants include:

1.  **ROUGE-N:** Measures the overlap of *n*-grams (sequences of *N* words).
    *   **ROUGE-1:** Counts matching single words (unigrams), assessing the presence of key content.
    *   **ROUGE-2:** Counts matching two-word sequences (bigrams), which gives a better indication of fluency and phrase accuracy.
2.  **ROUGE-L:** Based on the Longest Common Subsequence (LCS). Unlike n-grams, LCS doesn't require consecutive matches, making it more flexible to rephrasing and word order variations while still capturing sentence-level structure.

ROUGE scores are typically reported as precision, recall, or an F1-score (harmonic mean of precision and recall), providing a balanced view. While widely adopted for its objectivity and automation, ROUGE is a purely lexical metric. It struggles with semantic equivalence, synonyms, or paraphrases, meaning a summary conveying the same meaning using different words might receive a lower score. Despite these limitations, ROUGE remains a foundational benchmark for assessing textual generation quality.