import json
import numpy as np
import pandas as pd
import xml.etree.cElementTree as et
from sklearn.model_selection import StratifiedKFold


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


topics = topics[topics['correct_stance'].isin(['helpful', 'unhelpful'])]
topic_list = np.array(topics['topic_id'].tolist())
correct_stance_list = np.array(topics['correct_stance'].tolist())


topics_split = []
stratifiedKFold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
for index, (training_topics_idx, test_topics_idx) in enumerate(stratifiedKFold.split(topic_list, correct_stance_list)):
    test_topics = topic_list[test_topics_idx]
    topics_split.append(test_topics.tolist())

with open('../data/2019topics_split.json', 'w') as file:
    json.dump(topics_split, file)
