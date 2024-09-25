MAX_FAQ_POOL = 10

MILVUS_URI: str = "http://0.0.0.0:19530"
DATABASE_NAME: str = "rag_search"
DB_CONNECT: dict = {
    "uri": MILVUS_URI,
    "db_name": DATABASE_NAME
}