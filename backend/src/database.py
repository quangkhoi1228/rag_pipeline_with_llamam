import os

import psycopg2
from pymilvus import MilvusClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

milvus_db = MilvusClient(
    uri="http://localhost:19530"
)
# Kết nối tới PostgreSQL


def pg_create_connection():
    connection = psycopg2.connect(
        host="localhost",
        port="5432",
        user="postgres",
        password="123456",
        database="db_llm"
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
