import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.cElementTree as et
from pyserini.search import SimpleSearcher


# Set global variables
parser = argparse.ArgumentParser()
parser.add_argument('-q', '--qrels', help='Qrels name', choices=['2019qrels', '2021qrels'], required=True)
parser.add_argument('-i', '--index_path', help='Path to the index', required=True)
args = parser.parse_args()


# Function for converting HTML into plaintext
def get_content(html):
    try:
        start_index = html.index('<html')
        html = html[start_index:]
    except:
        pass
    soup = BeautifulSoup(html)
    text = soup.text
    text = re.sub('\r\n', '\n', text)
    text = re.sub(r'[\n]+', '\n', text)
    text = re.sub('\xa0', ' ', text)
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        processed_line = re.sub(r'\s+', ' ', line)
        if len(processed_line.strip()) > 0:
            processed_lines.append(processed_line.strip())
    text = '\n'.join(processed_lines)
    text = text.strip()
    text = text.encode('UTF-8', 'ignore').decode('UTF-8')
    return text


# Function for converting doc_id from the format specified by the TREC 2021 Health Misinformation Track
# to the original format
def convert(converted_doc_id):
    items = str(converted_doc_id).split('.')
    numbers = items[3].split('-')
    return f'c4-{int(numbers[0]):04d}-{int(items[4]):06d}'


# Start Retrieving
if args.qrels == '2019qrels':
    qrels = pd.read_csv('../data/raw_data/2019qrels_raw.txt', sep=' ', header=None, index_col=False,
                        names=['topic_id', 'iteration', 'doc_id', 'usefulness', 'stance', 'credibility'])
    qrels = qrels.drop(['iteration'], axis=1)
    qrels = qrels[qrels['stance'].isin([1, 3])]

    # Add queries and answers
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

    # Merge qrels and topics
    dataset = pd.merge(qrels, topics[['topic_id', 'query', 'description', 'correct_stance']],
                       how='left', on=['topic_id'])
    dataset['text'] = None
    dataset['timestamp'] = None
    dataset['url'] = None

    # Start retrieving
    searcher = SimpleSearcher(args.index_path)
    for line_index in tqdm(range(dataset.shape[0])):
        doc_id = dataset.loc[line_index, 'doc_id']
        try:
            doc = searcher.doc(doc_id)
            dataset.loc[line_index, 'text'] = get_content(doc.raw())
            dataset.loc[line_index, 'timestamp'] = doc.lucene_document().get('date')
            dataset.loc[line_index, 'url'] = doc.lucene_document().get('url')
        except:
            print(f'[ERROR] Document {doc_id} can not be found!')

elif args.qrels == '2021qrels':
    qrels = pd.read_csv('../data/raw_data/qrels-35topics.txt', sep=' ', header=None, index_col=False,
                        names=['topic_id', 'iteration', 'doc_id', 'usefulness', 'stance', 'credibility'])
    qrels = qrels.drop(['iteration'], axis=1)
    qrels = qrels[qrels['stance'].isin([0, 2])]

    # Add queries and answers
    xml_root = et.parse('../data/raw_data/misinfo-2021-topics.xml')
    rows = xml_root.findall('topic')
    xml_data = [[int(row.find('number').text), row.find('query').text, row.find('description').text,
                 row.find('narrative').text, row.find('stance').text, row.find('evidence').text] for row in rows]
    topics = pd.DataFrame(xml_data,
                          columns=['topic_id', 'query', 'description', 'narrative', 'correct_stance', 'evidence'])

    # Merge qrels and topics
    dataset = pd.merge(qrels, topics[['topic_id', 'query', 'description', 'correct_stance']],
                       how='left', on=['topic_id'])
    dataset['text'] = None
    dataset['timestamp'] = None
    dataset['url'] = None
    dataset['doc_id'] = dataset['doc_id'].apply(lambda x: convert(x))

    # Start retrieving
    searcher = SimpleSearcher(args.index_path)
    for line_index in tqdm(range(dataset.shape[0])):
        doc_id = dataset.loc[line_index, 'doc_id']
        doc_json = json.loads(searcher.doc(doc_id).raw())
        dataset.loc[line_index, 'text'] = doc_json['text']
        dataset.loc[line_index, 'timestamp'] = doc_json['timestamp']
        dataset.loc[line_index, 'url'] = doc_json['url']

else:
    raise ValueError('[ERROR] Invalid parameter "qrels".')


dataset.to_csv(f'../data/{args.qrels}_docs.csv', index=False)
