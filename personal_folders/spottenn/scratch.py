import requests

BASE_URL = "https://wwp-qdrant.spottenn.com"
COLLECTION_NAME = "wwp"
API_KEY = "<provide-your-own-key>"  # Replace with your actual API key
HEADERS = {
    'api-key': API_KEY,
    'Content-Type': 'application/json'
}


def fetch_all_vectors():
    # Define endpoint
    endpoint = f"{BASE_URL}/collections/{COLLECTION_NAME}/points/search"

    # Define payload for search (fetch all vectors and their payloads)
    data = {
        "params": {
            "vector": [0] * 512,  # Assuming vector dimension is 512, adjust as needed
            "top": 54441  # Total number of vectors
        }
    }

    response = requests.post(endpoint, headers=HEADERS, json=data)
    if response.status_code != 200:
        raise Exception("Failed to fetch vectors:", response.text)

    return response.json().get("points", [])


def delete_vectors(point_ids):
    # Define endpoint
    endpoint = f"{BASE_URL}/collections/{COLLECTION_NAME}/points/delete"

    # Define payload for delete
    data = {"ids": point_ids}

    response = requests.post(endpoint, headers=HEADERS, json=data)
    if response.status_code != 200:
        raise Exception("Failed to delete vectors:", response.text)

    return response.json()


# Fetch all vectors
vectors = fetch_all_vectors()

# Group by vector content and payload to find duplicates
duplicate_ids = []
seen = set()
for vector in vectors:
    vector_data = tuple(vector["vector"])
    payload_data = tuple(vector["payload"].items())
    unique_key = (vector_data, payload_data)

    if unique_key in seen:
        duplicate_ids.append(vector["id"])
    else:
        seen.add(unique_key)

# Delete duplicates
if duplicate_ids:
    delete_response = delete_vectors(duplicate_ids)
    print(f"Deleted {len(duplicate_ids)} duplicate vectors.")
else:
    print("No duplicates found.")
