from pymilvus.model.hybrid import BGEM3EmbeddingFunction  # type: ignore

embedding = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3',  # Specify the model name
    device='cpu',  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    use_fp16=False,
)


def embedding_document(docs):
    docs_embeddings = embedding.encode_documents(docs)
    vectors = docs_embeddings["dense"]
    return vectors


def embedding_query(query):
    query_embeddings = embedding.encode_queries([query])
    return query_embeddings['dense'][0]

def embedding_query2(query):
    query_embeddings = embedding.encode_queries([query])
    return query_embeddings