import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import xml.etree.cElementTree as et
from pyserini.search import SimpleSearcher


# Set global variables
parser = argparse.ArgumentParser()
parser.add_argument('-T', '--topic_set', help='Topic Set', choices=['2019', '2021', 'WH'], required=True)
parser.add_argument('-t', '--topic_id', help='Topic ID', required=True)
parser.add_argument('-i', '--index_path', help='Path to the index', required=True)
args = parser.parse_args()
k = 3000  # Retrieve top k documents per topic


# Load topics
if args.topic_set == '2019':
    xml_root = et.parse('../data/raw_data/2019topics.xml')
    rows = xml_root.findall('topic')
    xml_data = [[int(row.find('number').text), row.find('query').text, row.find('description').text,
                 row.find('narrative').text, row.find('cochranedoi').text] for row in rows]
    topics = pd.DataFrame(xml_data, columns=['topic_id', 'query', 'description', 'narrative', 'evidence'])
    topics_stance = pd.read_csv('../data/raw_data/2019topics_efficacy.txt',
                                header=None, sep=' ', names=['topic_id', 'correct_stance'])
    topics_stance['correct_stance'] = topics_stance['correct_stance'].map({1: 'helpful', 0: 'inconclusive',
                                                                           -1: 'unhelpful'})
    topics = pd.merge(topics, topics_stance, how='left', on='topic_id')
elif args.topic_set == '2021':
    xml_root = et.parse('../data/raw_data/misinfo-2021-topics.xml')
    rows = xml_root.findall('topic')
    xml_data = [[int(row.find('number').text), row.find('query').text, row.find('description').text,
                 row.find('narrative').text, row.find('stance').text, row.find('evidence').text] for row in rows]
    topics = pd.DataFrame(xml_data,
                          columns=['topic_id', 'query', 'description', 'narrative', 'correct_stance', 'evidence'])
elif args.topic_set == 'WH':
    k = 100  # Only the top 100 documents are needed
    topics = pd.read_csv('../data/WH_topics.csv')
    topics['description'] = 'none'


# Specify the topic
topics = topics[topics['topic_id'] == int(args.topic_id)]
topic_id = int(args.topic_id)
query = topics['query'].values.item()
description = topics['description'].values.item()
correct_stance = topics['correct_stance'].values.item()

# Start searching
searcher = SimpleSearcher(args.index_path)
searcher.set_bm25(0.9, 0.4)
hits = searcher.search(query, k=k)

records = []
for i in tqdm(range(k)):
    doc_id = hits[i].docid
    doc_json = json.loads(searcher.doc(doc_id).raw())
    records.append([topic_id, doc_id, i + 1, hits[i].score, query, description, correct_stance, doc_json['text'],
                    doc_json['timestamp'], doc_json['url']])

output_dir = Path(f'./output/{args.topic_set}')
output_dir.mkdir(parents=True, exist_ok=True)
dataset = pd.DataFrame(records,
                       columns=['topic_id', 'doc_id', 'rank', 'score', 'query', 'description', 'correct_stance', 'text',
                                'timestamp', 'url'])

dataset.to_csv(f'./output/{args.topic_set}/{args.topic_id}.csv', index=False)
