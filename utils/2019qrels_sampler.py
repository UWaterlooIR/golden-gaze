import numpy as np
import pandas as pd


dataset = pd.read_csv('../data/2019qrels_docs.csv')
dataset = dataset.dropna()
dataset.stance = dataset.stance.map({1: 0, 3: 1})
dataset = dataset.reset_index(drop=True)

topics = sorted(set(dataset.topic_id.values))

selected_indices = []
for topic in topics:
    dataset_part = dataset[dataset.topic_id == topic]
    stance_set = set(dataset_part.stance.values)
    if 0 in stance_set and 1 in stance_set:
        count = [dataset_part[dataset_part.stance == i].shape[0] for i in [0, 1]]
        minimum = min(count)
        for i in [0, 1]:
            sample_num = min(minimum, dataset_part[dataset_part.stance == i].shape[0])
            selected_part = dataset_part[dataset_part.stance == i].sample(sample_num, replace=False,
                                                                          random_state=42, axis=0)
            selected_indices.extend(selected_part.index.tolist())

selected_indices = np.array(sorted(selected_indices))
dataset = dataset.loc[selected_indices, :]

dataset.to_csv('../data/2019qrels_docs_balanced.csv', index=False)




