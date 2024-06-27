import requests
import json
from sklearn.metrics.pairwise import cosine_similarity

# Replace with your actual Google API key
google_api_key = 'AIzaSyCO5sgCM2vwMryARgmCgp6suac-_CmJXXA'

# API endpoint URL
url = f'https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={google_api_key}'


# Function to get embeddings
def get_embeddings(text):
    data = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text}]
        }
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, json=data, headers=headers)
    response_json = response.json()

    # Assuming the embedding vector is inside response_json['embedding']['values']
    embedding_vector = response_json['embedding']['values']
    return embedding_vector


# Example texts
texts = ["What is the meaning of life?", "How much wood would a woodchuck chuck?"]

# Get embeddings for each text
embeddings = [get_embeddings(text) for text in texts]

# Compute cosine similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"Similarity between text 1 and text 2: {similarity[0][0]}")
