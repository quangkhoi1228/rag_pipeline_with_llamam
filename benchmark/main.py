import os
from ragas.metrics import faithfulness, answer_correctness, answer_relevancy, answer_similarity, context_recall

from langchain_community.embeddings import CohereEmbeddings
from langchain_community.llms import Cohere

from src.benchmark import Benchmark


os.environ["COHERE_API_KEY"] = "qOTcu0Xr3BGmpAO8zk9z41y7gx06mO3cgGQOcvlt"

benchmark = Benchmark(
    llm=Cohere(model="command-xlarge"),
    embeddings=CohereEmbeddings(model="embed-multilingual-v3.0"),
    metrics=[faithfulness, answer_correctness, answer_relevancy, answer_similarity, context_recall],
    max_tokens=3500,
    calc_token=lambda str: len(str.split()) * 6,
    can_no_contexts = True
)

benchmark.run_full(
    data_filepath="benchmark/response_llm.csv",
    result_filepath="benchmark/result.jsonl"
)