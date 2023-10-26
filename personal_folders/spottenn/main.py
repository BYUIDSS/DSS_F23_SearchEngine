from dotenv import dotenv_values
import openai
import tiktoken
import qdrant_client as qc
import qdrant_client.http.models as qmodels
import uuid
import pandas as pd


MODEL = "text-embedding-ada-002"
MODEL_COST_PER_TOKEN = 0.0001 / 1000
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

    # Add vectors to collection
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=qmodels.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        ),
    )

def get_qd_vector(text):
    # TODO: add parameter for different collections
    response = client.scroll(
        collection_name=COLLECTION_NAME,
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


def embed_text(text):
    # TODO: refactor so that the check is done outside of this function
    qd_vector = get_qd_vector(text)
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


def calculate_cost(text="", encoding=tiktoken.encoding_for_model(MODEL), model_cost_per_token=MODEL_COST_PER_TOKEN): 
    num_tokens = len(encoding.encode(text))
    cost = num_tokens * model_cost_per_token
    return cost, num_tokens


def calculate_costs(df=None, columns=['Text Only Transcript'], model=MODEL, model_cost_per_token=MODEL_COST_PER_TOKEN):
    if df is not None:
        total_cost = 0
        total_num_tokens = 0
        for column in columns:
            for index, row in df.iterrows():
                if isinstance(row[column], str):
                    cost, num_tokens = calculate_cost(text=row[column], encoding=tiktoken.encoding_for_model(model), model_cost_per_token=model_cost_per_token)
                    total_cost += cost
                    total_num_tokens += num_tokens
        return total_cost, total_num_tokens
    return 0, 0


def display_cost(df):
    cost, num_tokens = calculate_costs(df=df)
    print("Num Tokens: ", num_tokens)
    print("Cost: ", cost)


if __name__ == '__main__':

    # Create index: Needs to be run once
    # create_index()
    
    # Read in the csv, display the cost, create embeddings, and store in Qdrant
    df = pd.read_csv('full-pages-export.csv')

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
