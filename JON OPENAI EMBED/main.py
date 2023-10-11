from dotenv import dotenv_values
import openai
import qdrant_client as qc
import qdrant_client.http.models as qmodels
import uuid
import pandas as pd

MODEL = "text-embedding-ada-002"
client = qc.QdrantClient(url="http://qdrant.orb.local/")
METRIC = qmodels.Distance.DOT
DIMENSION = 1536
COLLECTION_NAME = "wwp"


def create_subsection_vector(
        subsection_content,
        page_url,
):
    vector = embed_text(subsection_content)
    id = str(uuid.uuid1().int)[:32]
    payload = {
        "text": subsection_content,
        "url": page_url,

    }
    return id, vector, payload


def add_doc_to_index(page_content, page_url):
    ids = []
    vectors = []
    payloads = []

    id, vector, payload = create_subsection_vector(
        page_content,
        page_url,
    )
    ids.append(id)
    vectors.append(vector)
    payloads.append(payload)

    # Add vectors to collection
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=qmodels.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        ),
    )


def embed_text(text):
    config = dotenv_values(".env")
    openai.api_key = config['OPENAI_API_KEY']
    response = openai.Embedding.create(
        input=text,
        model=MODEL
    )
    embeddings = response['data'][0]['embedding']
    return embeddings


def create_index():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=DIMENSION,
            distance=METRIC,
        )
    )


def query_index(query, top_k=20, doc_types=None, block_types=None):
    vector = embed_text(query)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )

    results = [
        (
            f"{res.payload['url']}",
            res.payload["text"],
            res.score,
        )
        for res in results
    ]

    return results

if __name__ == '__main__':

    # Create index: Needs to be run once
    # create_index()
    #
    # Read in the csv, create embeddings, and store in Qdrant
    # df = pd.read_csv('pages-export-embeddings.csv')
    #
    # for index, row in df.iterrows():
    #     if isinstance(row['Text Only Transcript'], str):
    #         add_doc_to_index(
    #             row['Text Only Transcript'],
    #             row['Short URL'],
    #         )
    #     else:
    #         print(row['Text Only Transcript'])

    # Run search and print results
    results = query_index('When did you come to utah?')

    for result in results:
        print(result)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
