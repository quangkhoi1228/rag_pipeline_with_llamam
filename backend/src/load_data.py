import json
import sys
import time
import traceback
from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    connections,
    MilvusClient
)
import pandas as pd
from env_config import config
from model import DataUpload
import torch
import pickle
# from pqdm.processes import pqdm
from tqdm import tqdm

# list_qa = []
# with open('test_data/cleaned_tvpl.jsonl', 'r') as f:
#     for line in f :
#         data = json.loads(line)
#         if len(data['answers']) > 0:
#             data_upload = DataUpload(question = data['question'], answer = data['answers'][0], 
#                                     url = data['url'], post_time = data['post_time'])
#             list_qa.append(data_upload)

# print(len(list_qa))

# path_dense = "test_data/cleaned_tvpl_dense.pkl"

# with open(path_dense, 'rb') as f:
#     ques_embeddings = torch.tensor(pickle.load(f))

# print(ques_embeddings.shape)

from glob import glob
from scipy import sparse
import numpy as np

# folders = glob("../features/*")
folders = glob("./features/*")
folders.sort()
folders

print(folders)

def load_data(folder):
    df = pd.read_parquet(f"{folder}/encoded_data.parquet")
    title_sparse = sparse.load_npz(f"{folder}/title_sparse.npz")

    df["title_sparse"] = list(title_sparse)

    df = df.explode("content_text")
    content_sparse = sparse.load_npz(f"{folder}/content_sparse.npz")

    with open(f"{folder}/content_dense.npy", "rb") as file:
        content_dense = np.load(file)

    df["content_sparse"] = list(content_sparse)
    df["content_dense"] = list(content_dense)
    print(df.iloc[0:1])

    # print(df.iloc[0:1]['title_sparse'].values)
    # sys.exit()

    return df


def insert_to_collection(df, collection, batch_size=20):
    for i in tqdm(range(0, len(df), batch_size)):
        end_idx = min(len(df), i + batch_size)
        collection.insert(
            df.iloc[i : end_idx][
                [
                    "url",
                    "title_sparse",
                    "title_dense",
                    "title",
                    "content_sparse",
                    "content_dense",
                    "content_text",
                ]
            ]
            .rename(columns={"title": "title_text"})
            .to_dict("records")
        )

## Init milvus connection
connections.connect(**config.DB_CONNECT)

client = MilvusClient(**config.DB_CONNECT)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=8012),
    FieldSchema(name="title_dense", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="title_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="title_text", dtype=DataType.VARCHAR, max_length=8012),
    FieldSchema(name="content_dense", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="content_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=8012),
]

schema = CollectionSchema(fields=fields, enable_dynamic_field=False)

collection_name = "thu_vien_phap_luat"

# collection = Collection(name=collection_name, schema=schema, consistency_level="Bounded")
# collection.drop()
# sys.exit()

if not client.has_collection(collection_name=collection_name):
    collection = Collection(name=collection_name, schema=schema, consistency_level="Bounded")
    dense_index = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {
            "nlist": 1024
        }
    }
    collection.create_index("title_dense", dense_index)
    collection.create_index("content_dense", dense_index)

    sparse_index = {
        "index_type": "SPARSE_WAND",
        "metric_type": "IP",
    }
    collection.create_index("title_sparse", sparse_index)
    collection.create_index("content_sparse", sparse_index)

    collection.load()
else:
    collection = Collection(name=collection_name, schema=schema, consistency_level="Bounded")


for folder in folders:
    df = load_data(folder)
    print(folder, df.shape)
    insert_to_collection(df, collection)
    print("complete inserted df")
    print()
    break

# def index_data(data: list[DataUpload]):
#     try:
#         s_t = time.time() * 1000
#         questions: list[str] = list(map(lambda row: row.question, data))
#         answers = list(map(lambda row: row.answer, data))
#         post_times = list(map(lambda row: row.post_time, data))
#         urls = list(map(lambda row: row.url, data))

#         entities = [
#             questions,
#             answers,
#             post_times,
#             urls,
#             embeddings["dense_vecs"],
#             embeddings["lexical_weights"]
#         ]
#         collection.insert(data=entities)
#         collection.flush()
#         print(f"Indexed {len(data)} documents in {time.time() * 1000 - s_t} ms")
#     except Exception as e:
#         print.error(traceback.print_exc())
#         print.error(e)

# index_data(list_qa)
