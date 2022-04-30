import gc
import sys
import time
import json
import yaml
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.memory import garbage_collection_cuda
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report

import models
import utils

# Set global variables
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Name of the config file', required=True)
parser.add_argument('-t', '--topic', help='Topic ID', required=False)
args = parser.parse_args()

# Set config
with open(str(args.config) + '.yaml', 'r') as file:
    params = yaml.safe_load(file)

output_dir = Path('./output')
logs_dir = Path('logs/' + str(args.config) + '_logs/')
output_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)
params['prediction_save_path'] = f'./output/prediction_{args.config}'
params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pl_trainer_params = {
    'gpus': params['gpus'],
    'max_epochs': params['max_epochs'],
    'num_sanity_val_steps': params['num_sanity_val_steps'],
    'log_every_n_steps': params['log_every_n_steps'],
    'accumulate_grad_batches': params['accumulate_grad_batches']
}
tb_logger = pl_loggers.TensorBoardLogger(f'logs/{args.config}_logs/')
pl_trainer_params['logger'] = tb_logger
if params['mode'] == 'training':
    early_stop_callback = EarlyStopping(monitor='validation_f1', min_delta=0.001, patience=5, verbose=True, mode='max')
    pl_trainer_params['callbacks'] = [early_stop_callback]

random.seed(params['random_seed'])
np.random.seed(params['random_seed'])
torch.manual_seed(params['random_seed'])
torch.cuda.manual_seed_all(params['random_seed'])
seed_everything(params['random_seed'])

# Prepare the dataset
data = pd.read_csv(params['dataset_path'])
with open('../data/2019topics_split.json', 'r') as file:
    cross_validation_split_2019 = json.load(file)
topics = sorted(set(data.topic_id.values))
topics = np.array(topics, dtype='int')

# Start training
if params['mode'] == 'training':
    if params['evaluation'] == 'cross_validation':
        records = []
        for fold in range(5):
            training_topics = []
            for i in range(5):
                if i == fold:
                    continue
                training_topics.extend(cross_validation_split_2019[i])
            test_topics = cross_validation_split_2019[fold]

            training_topics = list(set(training_topics).intersection(set(data['topic_id'])))
            test_topics = list(set(test_topics).intersection(set(data['topic_id'])))

            print(f'[INFO] Fold {fold}: there are {len(training_topics)} training topics and '
                  f'{len(test_topics)} test topics.')

            training_topics, validation_topics = train_test_split(training_topics,
                                                                  test_size=0.1, random_state=params['random_seed'])

            training_dataset = utils.QrelDataset(params, data[data['topic_id'].isin(training_topics)])
            validation_dataset = utils.QrelDataset(params, data[data['topic_id'].isin(validation_topics)])
            test_dataset = utils.QrelDataset(params, data[data['topic_id'].isin(test_topics)])

            training_dataloader = DataLoader(training_dataset, batch_size=params['training_batch_size'], shuffle=True,
                                             num_workers=params['num_workers'])
            validation_dataloader = DataLoader(validation_dataset, batch_size=params['validation_batch_size'],
                                               shuffle=False,
                                               num_workers=params['num_workers'])
            test_dataloader = DataLoader(test_dataset, batch_size=params['validation_batch_size'], shuffle=False,
                                         num_workers=params['num_workers'])

            # Define and train the model.
            print()
            print('*-' * 5)
            print(f'* Fold {fold} *')
            print('*-' * 5)
            tb_logger = pl_loggers.TensorBoardLogger(f'logs/{args.config}_fold_{fold}_logs/')
            pl_trainer_params['logger'] = tb_logger
            if params['mode'] == 'training':
                early_stop_callback = EarlyStopping(monitor='validation_f1', min_delta=0.001, patience=5, verbose=True,
                                                    mode='max')
                pl_trainer_params['callbacks'] = [early_stop_callback]

            model = models.T5StanceDetector(params, training_dataset.tokenizer, data)
            trainer = pl.Trainer(**pl_trainer_params)
            trainer.fit(model, training_dataloader, validation_dataloader)
            trainer.save_checkpoint(f'output/{args.config}_fold{fold}.ckpt')
            print(f'[INFO] Model Saved at: output/{args.config}_fold{fold}.ckpt')
            trainer.test(model, test_dataloader)
            records.append(model.get_record())

            # Free useless memory
            garbage_collection_cuda()
            time.sleep(5)
            torch.cuda.empty_cache()
            garbage_collection_cuda()
            gc.collect()

        # Save results
        TPR = 0
        FPR = 0
        Acc = 0
        AUC = 0
        for i, record in enumerate(records):
            record['fold'] = i
            evaluation_results = utils.evaluate(record['helpful_probability'], record['target'])
            TPR += evaluation_results['TPR']
            FPR += evaluation_results['FPR']
            Acc += evaluation_results['accuracy']
            AUC += evaluation_results['AUC']
        prediction = pd.concat(records)
        prediction.to_csv(f'{params["prediction_save_path"]}.csv', index=False)
        print('*-' * 15)
        print('* Cross-Validation Performance *')
        print('*-' * 15)
        print(f'* TPR = {TPR / 5}')
        print(f'* FPR = {FPR / 5}')
        print(f'* Accuracy = {Acc / 5}')
        print(f'* AUC = {AUC / 5}')

    elif params['evaluation'] == 'training_only':
        training_topics, validation_topics = train_test_split(topics, test_size=0.1, random_state=params['random_seed'])
        training_dataset = utils.QrelDataset(params, data[data['topic_id'].isin(training_topics)])
        validation_dataset = utils.QrelDataset(params, data[data['topic_id'].isin(validation_topics)])
        training_dataloader = DataLoader(training_dataset, batch_size=params['training_batch_size'], shuffle=True,
                                         num_workers=params['num_workers'])
        validation_dataloader = DataLoader(validation_dataset, batch_size=params['validation_batch_size'],
                                           shuffle=False, num_workers=params['num_workers'])

        model = models.T5StanceDetector(params, training_dataset.tokenizer, data)
        trainer = pl.Trainer(**pl_trainer_params)
        trainer.fit(model, training_dataloader, validation_dataloader)
        trainer.save_checkpoint(f'output/{args.config}.ckpt')
        print(f'[INFO] Model Saved at: output/{args.config}.ckpt')
    else:
        raise ValueError('[ERROR] Invalid parameter "evaluation".')

elif params['mode'] == 'inference':
    inference_save_dir = Path(f'./output/{params["inference_save_dir"]}')
    inference_save_dir.mkdir(parents=True, exist_ok=True)

    if params['evaluation'] not in ['go_through', 'pre_defined_topic_split']:
        raise ValueError('[ERROR] Invalid parameter "evaluation".')

    if '2021qrels' in params['inference_dataset_prefix']:
        inference_data = pd.read_csv(params['inference_dataset_prefix'])
        inference_data = inference_data[inference_data.stance.isin([0, 2])]
        inference_data.stance = inference_data.stance.map({0: 0, 2: 1})
        inference_data = inference_data.reset_index(drop=True)

        inference_set = utils.QrelDataset(params, inference_data)
        inference_dataloader = DataLoader(inference_set, batch_size=params['inference_batch_size'], shuffle=False,
                                          num_workers=params['num_workers'])

        print(f'[INFO] Using saved model {params["checkpoint_path"]}')
        model = models.T5StanceDetector.load_from_checkpoint(f'output/{params["checkpoint_path"]}',
                                                             params=params, tokenizer=inference_set.tokenizer,
                                                             data=inference_data)
        trainer = pl.Trainer(**pl_trainer_params)
        trainer.test(model, inference_dataloader)
        record = model.get_record()
        evaluation_results = utils.evaluate(record['helpful_probability'], record['target'])
        print('*-' * 15)
        print('* Cross-Validation Performance *')
        print('*-' * 15)
        print(f'* TPR = {evaluation_results["TPR"]}')
        print(f'* FPR = {evaluation_results["FPR"]}')
        print(f'* Accuracy = {evaluation_results["accuracy"]}')
        print(f'* AUC = {evaluation_results["AUC"]}')
        record.to_csv(f'./output/{params["inference_save_dir"]}.csv')
        print(f'[INFO] Saved at: ./output/{params["inference_save_dir"]}.csv')

    elif params['evaluation'] == 'pre_defined_topic_split':
        inference_data = pd.read_csv(f'{params["inference_dataset_prefix"]}/{args.topic}.csv')
        inference_set = utils.QrelDataset(params, inference_data)
        inference_dataloader = DataLoader(inference_set, batch_size=params['inference_batch_size'], shuffle=False,
                                          num_workers=params['num_workers'])
        for fold in range(5):
            print(f'[INFO] Using saved model {params["checkpoint_path"]}_fold{fold}')
            model = models.T5StanceDetector.load_from_checkpoint(f'output/{params["checkpoint_path"]}_fold{fold}.ckpt',
                                                                 params=params,
                                                                 tokenizer=inference_set.tokenizer,
                                                                 data=inference_data)
            trainer = pl.Trainer(**pl_trainer_params)
            trainer.test(model, inference_dataloader)
            model.get_record().to_csv(f'./output/{params["inference_save_dir"]}/{args.topic}_fold_{fold}.csv')
            print(f'[INFO] Saved at: ./output/{params["inference_save_dir"]}/{args.topic}_fold_{fold}.csv')

    else:
        inference_data = pd.read_csv(f'{params["inference_dataset_prefix"]}/{args.topic}.csv')
        inference_set = utils.QrelDataset(params, inference_data)
        inference_dataloader = DataLoader(inference_set, batch_size=params['inference_batch_size'], shuffle=False,
                                          num_workers=params['num_workers'])
        model = models.T5StanceDetector.load_from_checkpoint(f'output/{params["checkpoint_path"]}', params=params,
                                                             tokenizer=inference_set.tokenizer,
                                                             data=inference_data)
        print(f'[INFO] Using saved model: {params["checkpoint_path"]}.')
        trainer = pl.Trainer(**pl_trainer_params)
        trainer.test(model, inference_dataloader)
        model.get_record().to_csv(f'./output/{params["inference_save_dir"]}/{args.topic}.csv')
        print(f'[INFO] Saved at: /output/{params["inference_save_dir"]}/{args.topic}.csv')

else:
    raise ValueError('[ERROR] Invalid parameter "mode".')
