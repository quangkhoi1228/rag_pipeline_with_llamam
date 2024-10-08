from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    connections,
    MilvusClient
)

## Search 
query = "hello world"

collection = Collection()