import math
import re
from typing import Dict, List, Tuple




def tokenize(text: str) -> List[str]:
    """Converts raw text into a normalized list of word tokens.

    This function performs two main processing steps:
    1. Lowercasing: Converts the entire string to lowercase so matching is
       case-insensitive (e.g., "Python" and "python" are treated identically).
    2. Tokenization: Uses the regular expression pattern r"[a-zA-Z0-9_]+" to
       extract sequences of alphanumeric characters and underscores as words,
       automatically ignoring punctuation and whitespace.

    Args:
        text (str): The raw string input to be tokenized.

    Returns:
        List[str]: A list of lowercase string tokens extracted from the input.

    Example:
        >>> tokenize("Hello, World! 123_test.")
        ['hello', 'world', '123_test']
    """
    # re.findall locates all non-overlapping matches of the regex pattern.
    # [a-zA-Z0-9_]+ matches any contiguous sequence of letters, digits, or underscores.
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())



# SEARCH ENGINE MODULE (TF-IDF + COSINE SIMILARITY)



def simple_tfidf_search(
    query: str,
    documents: List[Dict[str, str]],
    text_key: str = "content",
    top_k: int = 3,
) -> List[Dict[str, str]]:
    """Searches a collection of document dictionaries using TF-IDF weighting and

    Cosine Similarity.

    TF-IDF (Term Frequency - Inverse Document Frequency) measures how important a word
    is to a document within a collection:
    - Term Frequency (TF): How often a word appears in a specific document.
    - Inverse Document Frequency (IDF): How rare or common a word is across all documents.
      Rare words receive higher weight, while common words receive lower weight.

    Args:
        query (str): The search query string provided by the user.
        documents (List[Dict[str, str]]): A list of dictionaries representing documents.
            Each dictionary contains metadata and a text field.
        text_key (str, optional): The key in the document dictionaries where the text
            content is stored. Defaults to "content".
        top_k (int, optional): The maximum number of top-matching documents to return.
            Defaults to 3.

    Returns:
        List[Dict[str, str]]: A list of up to `top_k` document dictionaries, sorted by
            relevance score descending. Each dictionary is enriched with a new key,
            'similarity', containing the rounded cosine similarity score (0.0 to 1.0).

    Example:
        >>> docs = [{"id": 1, "content": "Python is fast"}, {"id": 2, "content": "Fast cars"}]
        >>> simple_tfidf_search("python", docs)
        [{'id': 1, 'content': 'Python is fast', 'similarity': 0.707}]
    """

   
    # STEP 1: Corpus Extraction & Tokenization
    
    # Extract raw text from each document using `text_key`. Default to empty string if missing.
    corpus = [doc.get(text_key, "") for doc in documents]

    # Convert each extracted raw text document into a list of lowercase tokens.
    # Resulting structure: List[List[str]] (e.g., [['python', 'is', 'fast'], ['fast', 'cars']])
    doc_tokens = [tokenize(t) for t in corpus]


  
    # STEP 2: Build Global Vocabulary
    
    # Flatten `doc_tokens` into a single set of unique terms across all documents,
    # then sort alphabetically to establish a fixed feature space.
    vocab = sorted(set(t for toks in doc_tokens for t in toks))

    # Guard Clause: If all documents are empty or contain no valid tokens, return an empty list.
    if not vocab:
        return []

    # STEP 3: Calculate Document Frequency (DF)
    
    # `df` maps each word in the vocabulary to the total number of documents containing that word.
    # This count is needed to compute the Inverse Document Frequency (IDF).
    df = {
        term: sum(1 for toks in doc_tokens if term in toks) for term in vocab
    }

    
    # STEP 4: Define Vectorization Helper Function (TF-IDF Vector Builder)

    def vec(tokens):
        """Converts a list of word tokens into a TF-IDF vector represented as a

        dictionary, where keys are vocabulary words and values are numerical TF-
        IDF weights.
        """
        # Count how many times each unique token appears in this specific token list (Term Frequency).
        counts = {t: tokens.count(t) for t in set(tokens)}

        # Total number of documents in the collection (N).
        n = len(doc_tokens)

        # Compute TF-IDF for every word in the global vocabulary:
        # - Term Frequency (TF): `counts.get(t, 0)`
        # - Inverse Document Frequency (IDF): `math.log((n + 1) / (df[t] + 1))`
        # Note: +1 smooths both numerator and denominator to prevent division-by-zero errors
        # and negative log values.
        return {
            t: counts.get(t, 0) * math.log((n + 1) / (df[t] + 1)) for t in vocab
        }

   
    # STEP 5: Vectorize the Search Query
    
    # Tokenize the user's query string and convert it into a TF-IDF feature vector.
    qv = vec(tokenize(query))

    # STEP 6: Define Cosine Similarity Function
    
    def cosine(a, b):
        """Calculates Cosine Similarity between two TF-IDF vectors `a` and `b`.

        Cosine Similarity measures the cosine of the angle between two vectors in a
        multidimensional space. Result ranges from 0.0 (completely orthogonal / un-related)
        to 1.0 (identical direction / perfectly matched terms).

        Formula:
            similarity = (a · b) / (||a|| * ||b||)
            where (a · b) is dot product, and ||a|| is Euclidean norm (magnitude).
        """
        # Dot Product: Sum of term-by-term multiplications across the vocabulary.
        dot = sum(a[t] * b[t] for t in vocab)

        # Vector Magnitudes (Euclidean Norms): Square root of the sum of squared weights.
        na = math.sqrt(sum(a[t] * a[t] for t in vocab))
        nb = math.sqrt(sum(b[t] * b[t] for t in vocab))

        # Avoid division by zero if either vector is all zeros (e.g., query has no matching vocab words).
        return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


    
    # STEP 7: Score & Rank Documents
    
    scored = []

    # Iterate over documents and their corresponding token lists side-by-side.
    for doc, toks in zip(documents, doc_tokens):
        # Calculate similarity score between query vector (qv) and document vector vec(toks).
        score = cosine(qv, vec(toks))
        scored.append((score, doc))

    # Sort scored tuple pairs in-place by score (index 0) in descending order (highest score first).
    scored.sort(key=lambda x: x[0], reverse=True)

    # STEP 8: Format and Return Top-K Results
    #
    results = []

    # Slice the sorted list to keep only the top `top_k` results.
    for score, doc in scored[:top_k]:
        # Make a shallow copy of the document dictionary to avoid modifying the caller's input data.
        d = dict(doc)

        # Attach the computed similarity score, rounded to 3 decimal places for readability.
        d["similarity"] = round(score, 3)

        results.append(d)

    return results