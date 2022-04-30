import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AdamW, T5ForConditionalGeneration
from sklearn.metrics import f1_score, classification_report


class T5StanceDetector(pl.LightningModule):
    def __init__(self, params, tokenizer, data):
        super(T5StanceDetector, self).__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.dataset = data
        self.model = T5ForConditionalGeneration.from_pretrained(params['pretrained_model_name'],
                                                                cache_dir=params['cache_dir'])
        self.mapping = [581, 4971]
        self.records = []

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def training_step(self, batch, batch_idx):
        lm_labels = batch['target_ids']
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['input_mask'],
                             labels=lm_labels, decoder_attention_mask=batch['target_mask'])

        logits = outputs.logits.clone().detach().cpu().numpy()
        logits = logits[:, 1, self.mapping]
        prediction = np.argmax(logits, axis=1).flatten()
        targets = batch['target'].to('cpu').numpy()

        self.log("training_loss", F.cross_entropy(torch.tensor(logits).float(), torch.tensor(targets).long()))
        return {'loss': outputs.loss, 'prediction': prediction, 'target': targets}

    def training_epoch_end(self, outputs):
        prediction = np.hstack([output['prediction'] for output in outputs])
        target = np.hstack([output['target'] for output in outputs])
        # print(f'\n{classification_report(target, prediction, zero_division=1)}\n')

    def validation_step(self, batch, batch_idx):
        outputs = self.model.generate(input_ids=batch['input_ids'],
                                      attention_mask=batch['input_mask'],
                                      max_length=self.params['max_output_sequence_length'],
                                      return_dict_in_generate=True, output_scores=True)

        logits = outputs.scores[0].clone().detach().cpu().numpy()
        logits = logits[:, self.mapping]
        prediction = np.argmax(logits, axis=1).flatten()
        targets = batch['target'].to('cpu').numpy()
        return {'prediction': prediction, 'target': targets}

    def validation_epoch_end(self, outputs):
        prediction = np.hstack([output['prediction'] for output in outputs])
        target = np.hstack([output['target'] for output in outputs])
        print('Epoch:', self.current_epoch)
        # print(f'\n{classification_report(target, prediction, zero_division=1)}\n')
        self.log("validation_f1", torch.tensor(f1_score(target, prediction, average='macro')).to(self.params['device']))

    def test_step(self, batch, batch_idx):
        outputs = self.model.generate(input_ids=batch['input_ids'],
                                      attention_mask=batch['input_mask'],
                                      max_length=self.params['max_output_sequence_length'],
                                      return_dict_in_generate=True, output_scores=True)

        logits = outputs.scores[0].clone().detach().cpu().numpy()
        logits = logits[:, self.mapping]
        prediction = np.argmax(logits, axis=1).flatten()
        targets = batch['target'].to('cpu').numpy()
        indices = batch['example_index'].to('cpu').numpy()
        logits = F.softmax(torch.tensor(logits), dim=1).numpy()

        for i in range(len(prediction)):
            bm25_score = -1
            if self.params['mode'] == 'inference' and 'qrels' not in self.params['inference_dataset_prefix']:
                bm25_score = self.dataset.loc[indices[i], 'score']
            self.records.append([self.dataset.loc[indices[i], 'topic_id'],
                                 self.dataset.loc[indices[i], 'correct_stance'],
                                 self.dataset.loc[indices[i], 'doc_id'], targets[i], prediction[i],
                                 bm25_score,
                                 logits[i, 0], logits[i, 1], self.dataset.loc[indices[i], 'url']])

        return {'prediction': prediction, 'target': targets}

    def test_epoch_end(self, outputs):
        print("\n*-*-*-*-*-*-*-*-*")
        print("* Test Result: *")
        print("*-*-*-*-*-*-*-*-*\n")
        prediction = np.hstack([output['prediction'] for output in outputs])
        target = np.hstack([output['target'] for output in outputs])
        print(f'\n{classification_report(target, prediction, zero_division=1)}\n')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.params['learning_rate'],
                          eps=1e-8
                          )
        return optimizer

    def get_record(self):
        return pd.DataFrame(self.records,
                            columns=['topic_id', 'correct_stance', 'doc_id', 'target', 'prediction',
                                     'bm25_score',
                                     'unhelpful_probability', 'helpful_probability', 'url'])
