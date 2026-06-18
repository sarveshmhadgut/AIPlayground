**ROUGE**, which stands for **Recall-Oriented Understudy for Gisting Evaluation**, is a widely used set of metrics designed to evaluate automatic text summarization and machine translation systems. It assesses the quality of a machine-generated summary by comparing it to one or more human-written reference summaries.

The metric works by measuring the overlap of words or phrases between the generated text and the reference text. The most common variants of ROUGE include:

*   **ROUGE-N:** Measures the overlap of $n$-grams. For example, ROUGE-1 calculates the overlap of single words (unigrams), while ROUGE-2 measures two-word sequences (bigrams).
*   **ROUGE-L:** Uses the Longest Common Subsequence (LCS) to find the longest shared sequence of words in structure, which naturally accounts for sentence-level word order.
*   **ROUGE-S:** Measures skip-bigram co-occurrence, allowing for words to be separated by arbitrary gaps.

ROUGE typically reports precision, recall, and F1-score. While precision measures how much of the generated summary is relevant, **recall** is highly emphasized because it determines how much of the essential information from the reference text was successfully captured. Despite its limitation in capturing synonyms, ROUGE remains an industry standard due to its simplicity and high correlation with human judgment.