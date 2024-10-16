import psycopg2
from pymilvus import (
    MilvusClient,
    WeightedRanker,
    RRFRanker,
    AnnSearchRequest,
    Collection,
    connections,
)
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

milvus_db = MilvusClient(uri="http://localhost:19530")

connections.connect(uri="http://localhost:19530")  # Replace with your Milvus server IP
collection = Collection(name="bat_dong_san")
collection.load()

rerank = RRFRanker()


# Kết nối tới PostgreSQL
def pg_create_connection():
    connection = psycopg2.connect(
        host="localhost",
        port="5432",
        user="postgres",
        password="123456",
        database="db_llm",
    )
    cursor = connection.cursor()
    return connection, cursor


def milvus_create_collection(collection_name, dimention=768):
    milvus_db.milvus_create_collection(
        collection_name=collection_name,
        dimension=dimention,  # The vectors we will use in this demo has 768 dimensions
    )


def milvus_delete_collection(collection_name):
    if milvus_db.has_collection(collection_name=collection_name):
        milvus_db.drop_collection(collection_name=collection_name)


def get_retrieve_context(query_embeds, limit=3):
    # query_embeds = bge_m3_ef.encode_queries([text])

    content_sparse_search_params = {
        "data": query_embeds["sparse"],
        "anns_field": "content_sparse",
        "param": {
            "metric_type": "IP",
        },
        "limit": limit,
    }
    content_sparse_request = AnnSearchRequest(**content_sparse_search_params)

    content_dense_search_params = {
        "data": query_embeds["dense"],
        "anns_field": "content_dense",
        "param": {
            "metric_type": "IP",
            "ef": 20,
        },
        "limit": limit,
    }
    content_dense_request = AnnSearchRequest(**content_dense_search_params)

    reqs = [
        content_sparse_request,
        content_dense_request,
    ]

    res = collection.hybrid_search(
        reqs,  # List of AnnSearchRequests created in step 1
        rerank,  # Reranking strategy specified in step 2
        limit=limit,  # Number of final search results to return
        output_fields=["id", "url", "title_text", "content_text"],
    )
    return res
