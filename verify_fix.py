from sentence_transformers import SentenceTransformer
import torch

def test_model():
    print("Loading model...")
    try:
        model = SentenceTransformer("embeddinggemma")
        print(model)
    except Exception as e:
        # Fallback if local path not found or issues with loading
        print(f"Error loading model directly: {e}")
        return

    sentences = [
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]

    print("Encoding sentences...")
    try:
        embeddings = model.encode(sentences)
        print(f"Embeddings shape: {embeddings.shape}")
        
        similarities = model.similarity(embeddings, embeddings)
        print("Similarities computed successfully:")
        print(similarities)
        print("\nSUCCESS: Model works independently!")
    except RuntimeError as e:
        print("\nFAILURE: RuntimeError encountered as expected (before fix):")
        print(e)
    except Exception as e:
        print(f"\nFAILURE: Unexpected error: {e}")

if __name__ == "__main__":
    test_model()
