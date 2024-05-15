import math
from collections import defaultdict


def tokenize(text):
    return text.lower().split()

def compute_tf(doc):
    tf = defaultdict(int)
    for word in doc:
        tf[word] += 1
    # Normalize term frequency by the number of words in the document
    for word in tf:
        tf[word] /= len(doc)
    return tf

def compute_idf(documents):
    idf = defaultdict(lambda: 0)
    total_documents = len(documents)
    for doc in documents:
        for word in set(doc):
            idf[word] += 1
    # Apply the IDF formula
    for word in idf:
        idf[word] = math.log(total_documents / (1 + idf[word]))
    return idf

def compute_tfidf(tf, idf):
    tfidf = defaultdict(float)
    for word in tf:
        tfidf[word] = tf[word] * idf[word]
    return tfidf

if __name__ == "__main__":

    # Sample documents
    documents = [
        "Python is a popular programming language.",
        "Machine learning algorithms can improve with more data.",
        "Artificial intelligence is transforming various industries.",
        "Data science involves statistics, machine learning, and data analysis.",
        "Neural networks are a key component of deep learning."
    ]

    # Step 1: Tokenize the Documents
    tokenized_documents = [tokenize(doc) for doc in documents]

    # Step 2: Compute Term Frequency (TF)
    tf_documents = [compute_tf(doc) for doc in tokenized_documents]

    # Step 3: Compute Inverse Document Frequency (IDF)
    idf = compute_idf(tokenized_documents)

    # Step 4: Compute TF-IDF
    tfidf_documents = [compute_tfidf(tf, idf) for tf in tf_documents]

    # Display TF-IDF Scores
    for i, tfidf in enumerate(tfidf_documents):
        print(f"Document {i+1} TF-IDF Scores:")
        for word, score in tfidf.items():
            print(f"  {word}: {score:.4f}")
        print()



