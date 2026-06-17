**ROUGE**, which stands for **Recall-Oriented Understudy for Gisting Evaluation**, is a widely used set of metrics in Natural Language Processing (NLP) designed to evaluate automatic text summarization and machine translation models. It works by comparing an automatically generated text (system summary) against one or more human-written reference texts.

ROUGE measures the overlap of words and phrases between the generated and reference texts. The most common variants include:

*   **ROUGE-N:** Measures the overlap of $n$-grams. For example, **ROUGE-1** looks at individual word matches (unigrams), while **ROUGE-2** looks at two-word phrase matches (bigrams).
*   **ROUGE-L:** Measures the Longest Common Subsequence (LCS). It identifies the longest shared sequence of words in the same order, which helps evaluate sentence-level structure and flow without requiring consecutive matches.
*   **ROUGE-S:** Evaluates skip-bigrams, which allows for gaps between matching words.

While originally "Recall-Oriented" to ensure the system captured all key information from the reference, modern ROUGE evaluations typically calculate **Recall** (how much reference content was captured), **Precision** (how much generated content was relevant), and the **F1-Score** (the balance between both). 

ROUGE remains a standard benchmark because it is fast, easy to compute, and correlates well with human judgment.