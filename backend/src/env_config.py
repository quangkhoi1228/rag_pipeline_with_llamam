from pydantic.v1 import BaseSettings


class Config(BaseSettings):
    MAX_FAQ_POOL = 10

    MILVUS_URI: str = "http://localhost:19530"
    DATABASE_NAME: str = "datltt_test"
    DB_CONNECT: dict = {
        "uri": MILVUS_URI,
        "db_name": DATABASE_NAME
    }
    MILVUS_COLLECTION: str = "thu_vien_phap_luat"

config: Config = Config()