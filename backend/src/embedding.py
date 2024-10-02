from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import torch

device= 'cuda' if torch.cuda.is_available() else 'cpu'

embedding = BGEM3EmbeddingFunction(
    'BAAI/bge-m3',  # Specify the model name
    device=device,  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    use_fp16=False,
)


def embedding_document(docs):
    docs_embeddings = embedding.encode(docs)
    vectors = docs_embeddings["dense"]
    return vectors


def embedding_query(query):
    query_embeddings = embedding.encode([query])
    return query_embeddings['dense'][0]


def get_embeddings(text):
    res = embedding.encode(text)
    return res