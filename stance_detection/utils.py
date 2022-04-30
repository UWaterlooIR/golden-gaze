import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from sklearn.metrics import roc_curve, auc


# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


class QrelDataset(Dataset):
    def __init__(self, params, data):
        super(QrelDataset, self).__init__()
        self.params = params
        self.data = data.copy(deep=True).reset_index(drop=False)

        # Set up the tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(params['tokenizer_name'], cache_dir=params['cache_dir'])

        self.porter_stemmer = PorterStemmer()
        self.words_of_interest = ['help', 'treat', 'benefit', 'effective', 'safe', 'improve', 'useful', 'reliable',
                                  'evidence',
                                  'prove', 'experience', 'find', 'conclude',
                                  'ineffective', 'harm', 'hurt', 'useless', 'limit', 'insufficient', 'dangerous', 'bad']
        self.words_of_interest = [self.porter_stemmer.stem(token) for token in self.words_of_interest]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        text = str(self.data.loc[index, 'text'])
        query = self.data.loc[index, 'query']
        if self.params['mode'] == 'inference' and 'qrels' not in self.params['inference_dataset_prefix']:
            target = 0
        else:
            target = self.data.loc[index, 'stance']

        preprocessed_text = self.preprocess(query, text)
        input_text = 'stance detection target : ' + query + ' document : ' + preprocessed_text

        tokenized_input = self.tokenizer.encode_plus(input_text.lower().strip(),
                                                     max_length=self.params['max_input_sequence_length'],
                                                     padding='max_length', truncation=True, return_attention_mask=True,
                                                     return_tensors='pt')

        target_mapping = {0: 'against', 1: 'favor'}

        tokenized_target = self.tokenizer.encode_plus(target_mapping[target],
                                                      max_length=self.params['max_output_sequence_length'],
                                                      padding='max_length', truncation=True, return_tensors='pt')

        return {'example_index': self.data.loc[index, 'index'], 'target': target,
                'input_ids': tokenized_input['input_ids'].squeeze(),
                'input_mask': tokenized_input['attention_mask'].squeeze(),
                'target_ids': tokenized_target['input_ids'].squeeze(),
                'target_mask': tokenized_target['attention_mask'].squeeze()}

    def preprocess(self, query: str, text: str):
        # against 581, favor 4971
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        pieces = []
        for line in text.split('\n'):
            for sentence in nltk.sent_tokenize(line):
                sentence = sentence.lower().strip()
                pieces.append(sentence)
        scores = []
        word_count = []
        query_tokens = [self.porter_stemmer.stem(query_token.lower()) for query_token in nltk.word_tokenize(query) if
                        query_token.lower() not in stop_words]

        processed_pieces = []
        for piece in pieces:
            piece = re.sub(r'[^A-Za-z]', ' ', piece)
            piece = re.sub(r'\s+', ' ', piece)
            piece = piece.lower().strip()
            processed_pieces.append(piece)

            words = nltk.word_tokenize(piece)
            word_count.append(len(words))
            score = 0
            for word in words:
                word = self.porter_stemmer.stem(word)
                if word in self.words_of_interest or word in query_tokens:
                    score += 1
            scores.append(score)

        scores = np.array(scores)
        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = []

        total_word_count = 0
        for idx in sorted_indices:
            if scores[idx] <= 0:
                break
            if total_word_count > self.params['max_input_sequence_length']:
                break
            if word_count[idx] >= self.params['min_sentence_words']:
                selected_indices.append(idx)
                total_word_count += word_count[idx]

        if total_word_count < self.params['max_input_sequence_length']:
            idx = min(selected_indices) if len(selected_indices) > 0 else 0
            while idx < len(scores):
                if idx in selected_indices:
                    idx += 1
                    continue
                if total_word_count > self.params['max_input_sequence_length']:
                    break
                if word_count[idx] >= self.params['min_sentence_words']:
                    selected_indices.append(idx)
                    total_word_count += word_count[idx]
                idx += 1

        selected_indices.sort()
        preprocessed_text = ' '.join(processed_pieces[index] for index in selected_indices)

        return preprocessed_text


def evaluate(y_prob, y_target):
    y_pred = np.where(y_prob >= 0.5, 1, 0)

    false_positive = np.sum((y_pred == 1) & (y_target == 0))
    true_positive = np.sum((y_pred == 1) & (y_target == 1))
    false_negative = np.sum((y_pred == 0) & (y_target == 1))
    true_negative = np.sum((y_pred == 0) & (y_target == 0))

    true_positive_rate = true_positive / (true_positive + false_negative)
    false_positive_rate = false_positive / (false_positive + true_negative)
    acc = (true_positive + true_negative) / len(y_pred)

    fpr, tpr, thresholds = roc_curve(y_target, y_prob)
    auc_value = auc(fpr, tpr)
    return {'TPR': true_positive_rate, 'FPR': false_positive_rate, 'accuracy': acc, 'AUC': auc_value}