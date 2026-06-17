**ROUGE**, which stands for **Recall-Oriented Understudy for Gisting Evaluation**, is a widely used set of metrics designed to evaluate automatic text summarization and machine translation. It works by comparing an automatically generated summary (the "system" summary) against one or more human-written reference summaries. 

Instead of analyzing semantic meaning, ROUGE measures the overlap of words and phrases between the generated text and the reference. The most common variants include:

*   **ROUGE-N:** Measures the overlap of $N$-grams. For example, ROUGE-1 looks at individual words (unigrams), while ROUGE-2 measures consecutive word pairs (bigrams).
*   **ROUGE-L:** Finds the Longest Common Subsequence (LCS) between the texts, which naturally accounts for word order and sentence structure without requiring consecutive matches.
*   **ROUGE-S:** Evaluates skip-bigrams, which are pairs of words in their sentence order, allowing for gaps between them.

For each variant, ROUGE calculates **Recall** (how much of the reference summary was captured), **Precision** (how much of the generated summary was relevant), and the **F1-Score** (the harmonic mean of both). While highly efficient and objective, ROUGE’s main limitation is that it relies on exact word matches, meaning it can penalize high-quality, creative paraphrasing.
