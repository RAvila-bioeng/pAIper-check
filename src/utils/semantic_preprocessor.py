# src/utils/semantic_preprocessor.py
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a pre-trained model. We use a lightweight one for efficiency.
# The first time this runs, it will download the model (approx. 90MB).
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_critical_snippets(text: str, anchor_text: str, num_snippets: int = 5) -> str:
    """
    Extracts the most semantically relevant snippets from a text in relation to an anchor text.

    Args:
        text (str): The main body of text to search within.
        anchor_text (str): The text to compare against (e.g., abstract, section title).
        num_snippets (int): The number of top snippets to return.

    Returns:
        str: A concatenated string of the most relevant snippets.
    """
    if not text or not anchor_text:
        return ""

    # 1. Divide the text into smaller, manageable chunks (e.g., paragraphs)
    chunks = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 10]
    if not chunks:
        return text[:1500] # Fallback for texts without clear paragraphs

    # 2. Convert anchor text and chunks into embeddings
    anchor_embedding = model.encode(anchor_text, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # 3. Calculate cosine similarity between the anchor and each chunk
    cosine_scores = util.pytorch_cos_sim(anchor_embedding, chunk_embeddings)[0]

    # 4. Find the indices of the top N most similar chunks
    # We use argpartition to find the top N without fully sorting the array
    
    # Ensure we don't request more snippets than available chunks to prevent errors
    if num_snippets > len(chunks):
        num_snippets = len(chunks)

    top_k_indices = np.argpartition(cosine_scores.cpu(), -num_snippets)[-num_snippets:]

    # 5. Retrieve the most relevant chunks and sort them by their original order
    top_chunks = [chunks[i] for i in sorted(top_k_indices)]

    # 6. Concatenate and return the critical snippets
    return "\n\n".join(top_chunks)

if __name__ == '__main__':
    # Example Usage
    sample_abstract = "This paper introduces a new method for deep learning-based image recognition, focusing on convolutional neural networks (CNNs). Our key contribution is a novel attention mechanism."
    
    sample_introduction = """
    Image recognition is a cornerstone of modern AI. Previous works have extensively used CNNs.
    However, these models often struggle with fine-grained classification.
    Our work proposes a new attention mechanism to address this limitation. We believe this will significantly improve performance.
    This paper will detail the architecture and experimental results. The goal is to set a new state-of-the-art.
    """

    print("--- Testing Semantic Snippet Extraction ---")
    
    critical_snippets = extract_critical_snippets(sample_introduction, sample_abstract, num_snippets=2)
    
    print(f"\n**Anchor Text (Abstract):**\n{sample_abstract}\n")
    print(f"**Original Text (Introduction):**\n{sample_introduction}\n")
    print(f"**Extracted Critical Snippets:**\n{critical_snippets}")

    # Expected output should contain the two sentences most related to the abstract:
    # "Our work proposes a new attention mechanism to address this limitation. We believe this will significantly improve performance."
    # "This paper will detail the architecture and experimental results. The goal is to set a new state-of-the-art."
