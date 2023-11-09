from dotenv import dotenv_values
import openai
import tiktoken
import qdrant_client as qc
import qdrant_client.http.models as qmodels
import uuid
import pandas as pd
# from estimate_costs.py import *


MODEL = "text-embedding-ada-002"
# QDRANT_URL = "http://qdrant.orb.local" 
QDRANT_URL = "https://wwp-qdrant.spottenn.com/"
config = dotenv_values(".env")
client = None
if "QDRANT_API_KEY" in config and config["QDRANT_API_KEY"] != "":
    client = qc.QdrantClient(url=QDRANT_URL,api_key=config['QDRANT_API_KEY'],port=443)
else:
    client = qc.QdrantClient(url=QDRANT_URL)
METRIC = qmodels.Distance.DOT
DIMENSION = 1536
COLLECTION_NAME = "wwp"
QUERY_COLLECTION_NAME = "wwp_query"



def upsert_points(collection_name, points):
    for points in points.values():
        get_from_cache(payload["text"], collection_name=collection_name)
        
    client.upsert(collection_name=collection_name, points=points)
    return


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
    # TODO: make it so it doesn't create duplicates
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

    points = qmodels.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )
    # points.
    # Add vectors to collection Commented out for now to prevent accidental duplicates
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    

def get_from_cache(text, collection_name=COLLECTION_NAME):
    # Check if the document/wuery is already in the database
    response = client.scroll(
        collection_name=collection_name,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="text",
                    match=qmodels.MatchValue(value=text)
                )
            ]
        ),
        with_vectors=True,
        limit=1
    )
    if len(response) > 0 and len(response[0]) > 0:
        return response[0][0].vector
    return None


def embed_text(text, collection_name=COLLECTION_NAME):
    # Check vector database before calling openai api
    qd_vector = get_from_cache(text, collection_name=collection_name)
    if qd_vector is not None:
        return qd_vector
    
    openai.api_key = config['OPENAI_API_KEY']
    response = openai.Embedding.create(
        input=text,
        model=MODEL
    )
    embeddings = response['data'][0]['embedding']
    return embeddings


def create_index():
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=DIMENSION,
            distance=METRIC,
        )
    )


def query_index(query, top_k=20, doc_types=None, block_types=None):
    vector = embed_text(query, collection_name=QUERY_COLLECTION_NAME)

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
    
    # Read in the csv, display the cost, create embeddings, and store in Qdrant
    # df = pd.read_csv('full-pages-export.csv')

    # display_cost(df)

    # for index, row in df.iterrows():
    #     if isinstance(row['Text Only Transcript'], str):
    #         add_doc_to_index(
    #             row['Text Only Transcript'],
    #             row['Short URL']
    #         )
    #     else:
    #         print(row['Text Only Transcript'], " ", row['Short URL'])

    # Run search and print results
    results = query_index('When did you come to utah?')


    for result in results:
        print(result)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
