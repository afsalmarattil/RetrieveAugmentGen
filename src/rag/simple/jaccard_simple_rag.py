"""
 Tokenization is the process by which we split a string into a list of "tokens" or words. 
 
"""

def tokenize(text):
    return set(text.lower().split())

"""
Jaccard Similarity is a simple way to measure how similar two sets are. Imagine each document as a collection of unique words. To find the Jaccard Similarity, you compare the common words in both documents to all the unique words from both documents. The formula is:

Jaccard Similarity = Number of common words / Total unique words
This gives you a number between 0 and 1, where 0 means no similarity and 1 means they are exactly the same.
"""

def jaccard_similarity(query, document):
    intersection = query.intersection(document)
    union = query.union(document)
    return len(intersection) / len(union)

def retrieve(query, documents, top_n=3):
    tokenized_query = tokenize(query)
    similarities = [(doc_id, doc, jaccard_similarity(tokenized_query, tokenize(doc))) for doc_id, doc in documents]
    similarities.sort(key=lambda x: x[2], reverse=True)
    return [(doc_id, doc) for doc_id, doc, sim in similarities[:top_n]]

def augment_and_generate(query, retrieved_docs):
    augmented_text = " ".join([doc for doc_id, doc in retrieved_docs])
    return f"Query: {query}\nAnswer: {augmented_text}"


if __name__ == "__main__":
    
    documents = [
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

    query = "How is AI transforming industries?"
    retrieved_docs = retrieve(query, documents)
    augmented_query = augment_and_generate(query, retrieved_docs)

    print("Retrieved Documents:", retrieved_docs)
    print("Augmented Query:", augmented_query)