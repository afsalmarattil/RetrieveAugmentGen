from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Tuple

def retrieve(query: str, documents: List[Tuple[int, str]], top_n: int = 3) -> List[Tuple[int, str, float]]:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc for _, doc in documents])
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return [(documents[index][0], documents[index][1], cosine_similarities[index]) for index in related_docs_indices]

def augment_and_generate(query: str, retrieved_docs: List[Tuple[int, str, float]]) -> str:
    augmented_text = " ".join([doc for index, doc, score in retrieved_docs])
    return f"Query: {query}\nAnswer: {augmented_text}"

if __name__ == "__main__":
    documents: List[Tuple[int, str]] = [
        (1, "Python is a popular programming language."),
        (2, "Machine learning algorithms can improve with more data."),
        (3, "Artificial intelligence is transforming various industries."),
        (4, "Data science involves statistics, machine learning, and data analysis."),
        (5, "Neural networks are a key component of deep learning."),
        (6, "Big data technologies are essential for handling large datasets."),
        (7, "Natural language processing enables computers to understand human language."),
        (8, "Computer vision allows machines to interpret and understand visual information."),
        (9, "Robotics integrates AI to perform complex tasks."),
        (10, "The Internet of Things (IoT) connects various devices to the internet.")
    ]

    query: str = "How is AI transforming industries?"
    retrieved_docs: List[Tuple[int, str, float]] = retrieve(query, documents)
    augmented_query: str = augment_and_generate(query, retrieved_docs)

    print("Retrieved Documents:", retrieved_docs)
    print("Augmented Query:", augmented_query)
