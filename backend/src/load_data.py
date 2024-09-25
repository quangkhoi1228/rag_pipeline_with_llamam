import json
import time
import traceback
from src.embedding import get_embeddings
from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    connections,
    MilvusClient
)

from src.config import DB_CONNECT
from src.model import DataUpload

list_qa = []
with open('test_data/cleaned_tvpl.jsonl', 'r') as f:
    for line in f :
        data = json.loads(line)
        data_upload = DataUpload(question = data['question'], answer = data['answers'][0], 
                                 url = data['url'], post_time = data['post_time'])
        list_qa.append(data_upload)

print(len(list_qa))


connections.connect(**DB_CONNECT)

client = MilvusClient(**DB_CONNECT)
collection_name = "test_rag"

fields = [
    FieldSchema(
        name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
    ),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=10240),
    FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="post_time", dtype=DataType.INT64),
    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1500),
    FieldSchema(name="ques_dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="ques_sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
]
schema = CollectionSchema(fields, "Search data for RAG", enable_dynamic_field=False)

if not client.has_collection(collection_name=collection_name):
    collection = Collection(name=collection_name, schema=schema, consistency_level="Bounded")
    dense_index = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {
            "nlist": 1024
        }
    }
    sparse_index = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP"
    }

    collection.create_index("ques_dense_vector", dense_index)
    collection.create_index("ques_sparse_vector", sparse_index)
    collection.load()
else:
    collection = Collection(name=collection_name, schema=schema, consistency_level="Bounded")

def index_data(data: list[DataUpload]):
    try:
        s_t = time.time() * 1000
        questions: list[str] = list(map(lambda row: row.question, data))
        embeddings = get_embeddings(questions)
        answers = list(map(lambda row: row.answer, data))
        post_times = list(map(lambda row: row.post_time, data))
        urls = list(map(lambda row: row.url, data))

        entities = [
            questions,
            answers,
            post_times,
            urls,
            embeddings["dense_vecs"],
            embeddings["lexical_weights"]
        ]
        collection.insert(data=entities)
        collection.flush()
        print(f"Indexed {len(data)} documents in {time.time() * 1000 - s_t} ms")
    except Exception as e:
        print.error(traceback.print_exc())
        print.error(e)

index_data()
