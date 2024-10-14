import json
import os
from typing import Dict, TypedDict, List, Callable


from cohere import Dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

import pandas as pd
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
        metrics: list[Metric],
        max_tokens: int = -1,
        calc_token: Callable[[str], int] = lambda _: 0,
        can_no_contexts: bool = False
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.metrics = metrics
        self.max_tokens = max_tokens
        self.calc_token = calc_token
        self.can_no_contexts = can_no_contexts

    def valid(self, data: QAData) -> bool:
        if 'question' not in data or 'ground_truth' not in data or 'answer' not in data or 'contexts' not in data:
            return False

        if self.max_tokens <= 0:
            return True

        left_tokens = self.max_tokens - (self.calc_token(data["question"]) + self.calc_token(
            data["ground_truth"]) + self.calc_token(data["answer"]))
        
        if left_tokens < 0:
            return False

        index = 0
        while index < len(data['contexts']):
            tokens = self.calc_token(data['contexts'][index])
            if tokens > left_tokens:
                break

            left_tokens -= tokens
            index += 1

        data['contexts'] = data['contexts'][:index]
        
        if self.can_no_contexts:
            if len(data['contexts']) == 0:
                data['contexts'] = [""]
            
            return True
        
        return len(data['contexts']) > 0

    def run_full(
        self,
        data_filepath: str,
        result_filepath: str,
        start_index: int = 0,
        end_index: int = -1,
    ):
        dataframe = pd.read_csv(data_filepath)
        results = Benchmark.load_result(result_filepath)

        def contexts_extract(text: str):
            return [i.strip() for i in text[1:-1].split("\n")]

        start_index = start_index if len(results) == 0 else (dataframe.index[(
            dataframe['question'] == results[-1]['question']) & (dataframe['ground_truth'] == results[-1]['ground_truth'])].tolist()[0] + 1)
        end_index = dataframe.shape[0] if end_index == - \
            1 else min(dataframe.shape[0], end_index)

        with open(result_filepath, mode="a", encoding="utf-8") as file:
            for i in tqdm(range(start_index, end_index), desc="Benchmark"):
                result: QAData = {
                    "question": dataframe.iloc[i]["question"],
                    "ground_truth": dataframe.iloc[i]["ground_truth"],
                    "answer": dataframe.iloc[i]["llm_answer"],
                    "contexts": contexts_extract(dataframe.iloc[i]["context"])
                }

                if not self.valid(result):
                    continue

                scores = self.evaluate(result)
                result.update(scores)

                results.append(result)

                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()

        return results

    def get_valid(
        self,
        data_filepath: str
    ):
        dataframe = pd.read_csv(data_filepath)

        def contexts_extract(text: str):
            return [i.strip() for i in text[1:-1].split("\n")]

        result = []
        for i in range(dataframe.shape[0]):
            data: QAData = {
                "question": dataframe.iloc[i]["question"],
                "ground_truth": dataframe.iloc[i]["ground_truth"],
                "answer": dataframe.iloc[i]["llm_answer"],
                "contexts": contexts_extract(dataframe.iloc[i]["context"])
            }

            if not self.valid(data):
                continue
            
            result.append(data)
        return result
            

    def run_with_providers(
        self,
        groundtruth_filepath: str,
        result_filepath: str,
        # inputs: question, list contexts
        answer_provider: Callable[[str, List[str]], str],
        contexts_provider: Callable[[str], List[str]],  # inputs: question
        start_index: int = 0,
        end_index: int = -1
    ):
        groundtruth_datas = Benchmark.load_groundtruth(groundtruth_filepath)
        results = Benchmark.load_result(result_filepath)

        start_index = start_index if len(results) == 0 else (next((i for i, x in enumerate(
            groundtruth_datas) if x["question"] == results[-1]["question"] and x["ground_truth"] == results[-1]["ground_truth"]), None) + 1)
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

                if not self.valid(result):
                    continue

                scores = self.evaluate(result)
                result.update(scores)

                results.append(result)

                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()

        return results

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

            if not (question and ground_truth):
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
