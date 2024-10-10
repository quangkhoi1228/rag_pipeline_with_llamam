import json
import os
from typing import Dict, TypedDict, List, Callable


from cohere import Dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.metrics.base import Metric

from datasets import Dataset
from tqdm import tqdm


class QAData(TypedDict):
    question: str
    ground_truth: str
    answer: str
    contexts: List[str]


class Benchmark:
    def __init__(
        self,
        llm: BaseRagasLLM | BaseLanguageModel,
        embeddings: BaseRagasEmbeddings | Embeddings,
        metrics: list[Metric]
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.metrics = metrics

    def run(
        self,
        groundtruth_filepath: str,
        result_filepath: str,
        answer_provider: Callable[[str, List[str]], str],
        contexts_provider: Callable[[str], List[str]],
        start_index: int = 0,
        end_index: int = -1
    ):
        groundtruth_datas = Benchmark.load_groundtruth(groundtruth_filepath)
        results = Benchmark.load_result(result_filepath)

        start_index = len(results) + start_index
        end_index = len(groundtruth_datas) if end_index == - \
            1 else min(len(groundtruth_datas), end_index)

        with open(result_filepath, mode="a", encoding="utf-8") as file:
            for i in tqdm(range(start_index, end_index), desc="Benchmark"):
                groundtruth_data: QAData = groundtruth_datas[i]

                result: QAData = {**groundtruth_data}
                result["contexts"] = contexts_provider(
                    groundtruth_data["question"])
                result["answer"] = answer_provider(
                    groundtruth_data["question"], result["contexts"])

                scores = self.evaluate(result)
                result.update(scores)

                results.append(result)

                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()

        return result

    def evaluate(self, qa: QAData) -> Dict:
        dataset = Dataset.from_dict({
            'question': [qa["question"]],
            'answer': [qa["answer"]],
            'contexts': [qa["contexts"]],
            'ground_truth': [qa["ground_truth"]]
        })

        result = evaluate(dataset,
                          metrics=self.metrics,
                          llm=self.llm,
                          embeddings=self.embeddings).scores

        return result[0]

    @staticmethod
    def load_groundtruth(filepath: str) -> List[QAData]:
        raw_data = []

        # load raw qa data
        with open(filepath, mode="r", encoding="utf-8") as file:
            for line in file.readlines():
                raw_data.append(json.loads(line))

        # cleaning
        result = []
        for e in raw_data:
            question = e.get('question', '')
            ground_truth = e.get('answers', [{}])[0].get('body', '')

            if not question or not ground_truth:
                continue

            result.append(QAData(question=question, ground_truth=ground_truth))

        return result

    @staticmethod
    def load_result(filepath: str) -> List[QAData]:
        result = []

        if not os.path.exists(filepath):
            return result

        with open(filepath, mode="r", encoding="utf-8") as file:
            for line in file.readlines():
                result.append(json.loads(line))

        return result
