import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc


# Set global variables
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--topics', help='topic set', choices=['2019', '2021'], required=True)
args = parser.parse_args()
top_k = 100  # For each topic, only top k documents retrieved by BM25 will be used.
with open('../data/2019topics_split.json', 'r') as file:
    cross_validation_split_2019 = json.load(file)


def get_features(given_dataset, given_topics, hostnames_dict):
    return_X = []
    for topic_id in given_topics:
        record = np.zeros((len(hostnames_dict)))
        partial_dataset = given_dataset[given_dataset['topic_id'] == topic_id]

        for idx, row in partial_dataset.iterrows():
            # For documents from the same domain, we only keep the first one.
            hostname = row['hostname']
            if hostname not in hostnames_dict.keys():
                continue
            hostname_index = hostnames_dict[hostname]
            if record[hostname_index] == 0:
                helpful_probability = row['helpful_probability']
                record[hostname_index] = 2 * helpful_probability - 1

        return_X.append(record.tolist())
    return return_X


def get_mapping_dict(hostnames):
    mapping_dict = {}
    for hostname in hostnames:
        mapping_dict[hostname] = len(mapping_dict) if hostname not in mapping_dict.keys() else mapping_dict[hostname]
    return mapping_dict, dict([(v, k) for (k, v) in mapping_dict.items()])


def run(training_dataset, test_dataset, fold):
    training_dataset['hostname'] = training_dataset['url'].apply(lambda x: urlparse(x).netloc)
    training_topics = sorted(set(training_dataset['topic_id']))
    training_topics = np.array(training_topics, dtype='int')
    training_answers = [1 if training_dataset[training_dataset['topic_id'] == topic].head(1)['correct_stance'].values[
                                 0] == 'helpful' else 0
                        for topic in training_topics]
    training_answers = np.array(training_answers, dtype='int')

    test_dataset['hostname'] = test_dataset['url'].apply(lambda x: urlparse(x).netloc)
    test_topics = sorted(set(test_dataset['topic_id']))
    test_topics = np.array(test_topics, dtype='int')
    test_answers = [
        1 if test_dataset[test_dataset['topic_id'] == topic].head(1)['correct_stance'].values[0] == 'helpful' else 0
        for topic in test_topics]
    test_answers = np.array(test_answers, dtype='int')

    training_domains = set(training_dataset['hostname'])
    training_domains_dict, training_domains_dict_reversed = get_mapping_dict(training_domains)
    training_X = get_features(training_dataset, training_topics, training_domains_dict)
    training_y = training_answers
    test_X = get_features(test_dataset, test_topics, training_domains_dict)
    test_y = test_answers

    model = LogisticRegression(penalty='none')
    model.fit(training_X, training_y)
    output_dir = Path('./output')
    output_dir.mkdir(parents=True, exist_ok=True)
    model_weights = pd.DataFrame(
        {'domains': [training_domains_dict_reversed[i] for i in range(len(training_domains_dict_reversed))],
         'helpful_weights': model.coef_[0][:len(training_domains_dict)]})
    if args.topics == '2019':
        model_weights.to_csv(f'output/2019_LR_weights_fold{fold}.csv', index=False)
    elif args.topics == '2021':
        model_weights.to_csv('output/2021_LR_weights.csv', index=False)

    test_probability = model.predict_proba(test_X)[:, 1]
    test_prediction = model.predict(test_X)

    return test_topics, test_prediction, test_probability, test_y


def evaluate(dataframe):
    y_pred = dataframe['predicted_stance']
    y_target = dataframe['correct_stance']
    y_prob = dataframe['prediction_prob']

    false_positive = np.sum((y_pred == 1) & (y_target == 0))
    true_positive = np.sum((y_pred == 1) & (y_target == 1))
    false_negative = np.sum((y_pred == 0) & (y_target == 1))
    true_negative = np.sum((y_pred == 0) & (y_target == 0))

    true_positive_rate = true_positive / (true_positive + false_negative)
    false_positive_rate = false_positive / (false_positive + true_negative)
    acc = (true_positive + true_negative) / len(y_pred)
    fpr, tpr, thresholds = roc_curve(y_target, y_prob)
    auc_value = auc(fpr, tpr)

    return true_positive_rate, false_positive_rate, acc, auc_value


# Start training and testing
if args.topics == '2019':
    record_topics = []
    record_prediction = []
    record_prediction_prob = []
    record_labels = []
    record_fold = []
    for fold_id in range(5):
        training_dataset = []
        for topic in range(90):
            file_path = f'../stance_detection/output/WHtopics_inference_cv/{topic}_fold_{fold_id}.csv'
            dataframe = pd.read_csv(file_path)
            dataframe.sort_values(['bm25_score'], ascending=False, inplace=True)
            dataframe = dataframe.head(top_k)
            training_dataset.append(dataframe)
        training_dataset = pd.concat(training_dataset, axis=0)

        test_topics = cross_validation_split_2019[fold_id]
        test_dataset = []
        for topic in sorted(test_topics):
            file_path = f'../stance_detection/output/2019topics_cv_inference/{topic}_fold_{fold_id}.csv'
            dataframe = pd.read_csv(file_path)
            dataframe.sort_values(['bm25_score'], ascending=False, inplace=True)
            dataframe = dataframe.head(top_k)
            test_dataset.append(dataframe)
        test_dataset = pd.concat(test_dataset, axis=0)

        test_topics, test_prediction, test_probability, test_y = run(training_dataset, test_dataset, fold_id)

        print('*-' * 6)
        print(f'* Fold: {fold_id} *')
        print('*-' * 6)
        print(classification_report(test_y, test_prediction, zero_division=1))
        fpr, tpr, threshold = roc_curve(test_y, test_probability)
        roc_auc = auc(fpr, tpr)
        print(f'AUC: {roc_auc}')

        record_topics.extend(test_topics)
        record_prediction_prob.extend(test_probability)
        record_prediction.extend(test_prediction)
        record_labels.extend(test_y)
        record_fold.extend([fold_id] * len(test_y))

    records = pd.DataFrame(
        [[record_fold[i], record_topics[i], record_prediction[i], record_prediction_prob[i], record_labels[i]]
         for i in range(len(record_topics))],
        columns=['fold', 'topic_id', 'predicted_stance', 'prediction_prob', 'correct_stance'])
    records.to_csv('output/2019_stance_prediction.csv', index=False)

    print('*-' * 12)
    print(f'* Overall Performance *')
    print('*-' * 12)
    print(classification_report(record_labels, record_prediction, zero_division=1))

    tpr_sum = 0
    fpr_sum = 0
    acc_sum = 0
    auc_sum = 0
    false_positive_rate = 0
    for i in range(5):
        part = records[records['fold'] == i]
        evaluation_results = evaluate(records)

        tpr_sum += evaluation_results[0]
        fpr_sum += evaluation_results[1]
        acc_sum += evaluation_results[2]
        auc_sum += evaluation_results[3]

    print(f'TPR: {tpr_sum / 5}')
    print(f'FPR: {fpr_sum / 5}')
    print(f'Accuracy: {acc_sum / 5}')
    print(f'AUC: {auc_sum / 5}')

elif args.topics == '2021':
    training_dataset = []
    for topic in range(90):
        file_path = f'../stance_detection/output/WHtopics_inference/{topic}.csv'
        dataframe = pd.read_csv(file_path)
        dataframe.sort_values(['bm25_score'], ascending=False, inplace=True)
        dataframe = dataframe.head(top_k)
        training_dataset.append(dataframe)
    training_dataset = pd.concat(training_dataset, axis=0)

    test_dataset = []
    for topic in range(101, 151):
        file_path = f'../stance_detection/output/2021topics_inference/{topic}.csv'
        dataframe = pd.read_csv(file_path)
        dataframe.sort_values(['bm25_score'], ascending=False, inplace=True)
        dataframe = dataframe.head(top_k)
        test_dataset.append(dataframe)
    test_dataset = pd.concat(test_dataset, axis=0)

    test_topics, test_prediction, test_probability, test_y = run(training_dataset, test_dataset, -1)

    records = pd.DataFrame([[test_topics[i], test_prediction[i], test_probability[i], test_y[i]]
                            for i in range(len(test_topics))],
                           columns=['topic_id', 'predicted_stance', 'prediction_prob', 'correct_stance'])
    records.to_csv('output/2021_stance_prediction.csv', index=False)

    print('*-' * 12)
    print(f'* Overall Performance *')
    print('*-' * 12)
    print(classification_report(test_y, test_prediction, zero_division=1))
    evaluation_results = evaluate(records)
    print(f'TPR: {evaluation_results[0]}')
    print(f'FPR: {evaluation_results[1]}')
    print(f'Accuracy: {evaluation_results[2]}')
    print(f'AUC: {evaluation_results[3]}')
