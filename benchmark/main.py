from typing import Dict, List
import json

import os
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_correctness, answer_relevancy
from datasets import Dataset
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# def load_qa_data(filepath: str) -> List[Dict]:
#     result = []
#     with open(filepath, mode="r", encoding="utf-8") as file:
#         for line in file.readlines():
#             result.append(json.loads(line))
        
#     return result

# def clean_qa_data(data: List[Dict]) -> List[Dict]:
#     return [{ 'question': e['question'], 'answer': e['answers'][0]['body'] } for e in data]

# qa_data = [load_qa_data("benchmark/thread_p180000-189999")[0]]
# cleaned_qa_data = clean_qa_data(qa_data)

critic_llm = ChatOllama(model="llama3.1")
critic_embedding = OllamaEmbeddings(model="llama3.1")

dataset = Dataset.from_dict({'question': ['What is the population of New York City as of 2020?'],
                'answer': ["According to the passage, New York City's population was 8,804,190 in 2020."],
                'contexts': [["New York City is the most populous city in the United States, with 8,804,190 residents incorporating more immigration into the city than outmigration since the 2010 United States census. More than twice as many people live in New York City as compared to Los Angeles, the second-most populous U.S. city; and New York has more than three times the population of Chicago, the third-most populous U.S. city. New York City gained more residents between 2010 and 2020 (629,000) than any other U.S. city, and a greater amount than the total sum of the gains over the same decade of the next four largest U.S. cities, Los Angeles, Chicago, Houston, and Phoenix, Arizona combined. New York City's population is about 44% of New York State's population, and about 39% of the population of the New York metropolitan area. The majority of New York City residents in 2020 (5,141,538, or 58.4%) were living on Long Island, in Brooklyn, or in Queens. The New York City metropolitan statistical area, has the",
                              "New York, often called New York City or NYC, is the most populous city in the United States. With a 2020 population of 8,804,190 distributed over 300.46 square miles (778.2 km2), New York City is the most densely populated major city in the United States and more than twice as populous as Los Angeles, the nation's second-largest city. New York City is located at the southern tip of New York State. It constitutes the geographical and demographic center of both the Northeast megalopolis and the New York metropolitan area, the largest metropolitan area in the U.S. by both population and urban area. With over 20.1 million people in its metropolitan statistical area and 23.5 million in its combined statistical area as of 2020, New York is one of the world's most populous megacities, and over 58 million people live within 250 mi (400 km) of the city. New York City is a global cultural, financial, entertainment, and media center with a significant influence on commerce, health care and life",
                              "=== Population density ===\n\nIn 2020, the city had an estimated population density of 29,302.37 inhabitants per square mile (11,313.71/km2), rendering it the nation's most densely populated of all larger municipalities (those with more than 100,000 residents), with several small cities (of fewer than 100,000) in adjacent Hudson County, New Jersey having greater density, as per the 2010 census. Geographically co-extensive with New York County, the borough of Manhattan's 2017 population density of 72,918 inhabitants per square mile (28,154/km2) makes it the highest of any county in the United States and higher than the density of any individual American city. The next three densest counties in the United States, placing second through fourth, are also New York boroughs: Brooklyn, the Bronx, and Queens respectively.\n\n\n=== Race and ethnicity ===",
                              "New York's population reached all-time highs in the 2000 census and then again in the 2010 census."],
                             ],
                'ground_truth': ['8,804,190']})

run_config = RunConfig(
    timeout=600,
    max_retries=3,
    max_wait=60,
    max_workers=1
)

# Run evaluation
score = evaluate(
    dataset,
    metrics=[faithfulness, answer_correctness, answer_relevancy],
    llm=critic_llm,
    embeddings=critic_embedding,
    run_config=run_config
)

print(score.to_pandas())
