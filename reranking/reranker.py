import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Set global variables
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tag', help='Run tag', choices=['BM25-Baseline', 'Trust-Pipeline', 'Correct-Stance'],
                    required=True)
args = parser.parse_args()
stance_path_map = {'BM25-Baseline': '../initial_retrieval/output/2021/',
                   'Trust-Pipeline': '../stance_detection/output/2021topics_inference/',
                   'Correct-Stance': '../stance_detection/output/2021topics_inference/', }

# Load files
prediction = []
for topic_id in range(101, 151):
    file_path = f'{stance_path_map[args.tag]}{topic_id}.csv'
    dataframe = pd.read_csv(file_path)
    prediction.append(dataframe)
prediction = pd.concat(prediction, axis=0)
topic_stance = pd.read_csv('../answer_prediction/output/2021_stance_prediction.csv')
prediction['combined_score'] = None
prediction['new_rank'] = None
prediction = prediction[~prediction['topic_id'].isin([113, 116, 119, 123, 124, 125, 126, 130, 135,
                                                      138, 141, 142, 147, 148, 150, 127, 133, 145])]
# Remove topics that only have correct judgments (127, 133, 145) and that were not judged.

topic_to_prob = {}
for idx, line in topic_stance.iterrows():
    topic_to_prob[line['topic_id']] = line['prediction_prob']


def combine_scores(record):
    if args.tag == 'BM25-Baseline':
        return record['score']
    topic_id = record['topic_id']
    bm25_score = record['bm25_score']
    correct_stance = record['correct_stance']
    unhelpful_probability = record['unhelpful_probability']
    helpful_probability = record['helpful_probability']
    if args.tag == 'Correct-Stance':
        correct_probability = helpful_probability if correct_stance == 'helpful' else unhelpful_probability
    elif args.tag == 'Trust-Pipeline':
        correct_probability = topic_to_prob[topic_id] * helpful_probability + \
                              (1 - topic_to_prob[topic_id]) * unhelpful_probability

    return bm25_score * np.exp(correct_probability - 0.5)


def fix_docno(record):
    docno = record['doc_id']
    if not re.match(r'c4-[0-9]+-[0-9]+', docno):
        raise SystemExit(f'\033[docnos are not in an acceptable string format: c4-[0-9]+-[0-9]+ !\033[0m')
    docno_split = docno.split('-')
    file_number = int(docno_split[1])
    json_loc = int(docno_split[2])
    return f'en.noclean.c4-train.{file_number:05d}-of-07168.{json_loc}'


prediction['combined_score'] = prediction.apply(combine_scores, axis=1)
prediction['doc_id'] = prediction.apply(fix_docno, axis=1)
prediction['iteration'] = 'Q0'
prediction['run_identifier'] = args.tag
prediction.sort_values(['topic_id', 'combined_score'], ascending=[True, False], inplace=True)
prediction = prediction.groupby(['topic_id']).head(1000)
prediction = prediction.reset_index(drop=True)
prediction['new_rank'] = np.concatenate([np.arange(1, 1001) for i in range(len(set(prediction['topic_id'])))])

prediction = prediction[['topic_id', 'iteration', 'doc_id', 'new_rank', 'combined_score', 'run_identifier']]
output_dir = Path('./output')
output_dir.mkdir(parents=True, exist_ok=True)
prediction.to_csv(f'output/{args.tag}', sep=' ', header=None, index=False)
