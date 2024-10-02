from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker
from embedding import embedding

rerank = RRFRanker()

def search(collection, text, limit=3):
    query_embeds = embedding.encode_queries([text])

    title_sparse_search_params = {
        "data": query_embeds["sparse"],  # Query vector
        "anns_field": "title_sparse",  # Vector field name
        "param": {
            "metric_type": "IP",  # This parameter value must be identical to the one used in the collection schema
        },
        "limit": limit,  # Number of search results to return in this AnnSearchRequest
    }
    title_sparse_request = AnnSearchRequest(**title_sparse_search_params)

    title_dense_search_params = {
        "data": query_embeds["dense"],
        "anns_field": "title_dense",
        "param": {
            "metric_type": "COSINE",
            "ef": 20,
        },
        "limit": limit,
    }
    title_dense_request = AnnSearchRequest(**title_dense_search_params)

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
            "metric_type": "COSINE",
            "ef": 20,
        },
        "limit": limit,
    }
    content_dense_request = AnnSearchRequest(**content_dense_search_params)

    reqs = [
        title_sparse_request,
        title_dense_request,
        content_sparse_request,
        content_dense_request,
    ]

    collection.load()
    res = collection.hybrid_search(
        reqs,  # List of AnnSearchRequests created in step 1
        rerank,  # Reranking strategy specified in step 2
        limit=limit,  # Number of final search results to return
        output_fields=["id", "title", "content_text", "url"],
    )
    return res


