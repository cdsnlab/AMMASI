#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import argparse
import time, tqdm
import mymodels
import utils
import warnings
import keras
from keras.callbacks import TensorBoard
import logging

slack_message = ""

from datetime import datetime 

start_time = datetime.now() 


# In[2]:


def model_define(args, metadata):
    num_features = metadata['num_features']

    X = layers.Input(shape=(num_features,), dtype=tf.float32) # X_feature
    
    tmp_model = mymodels.str_to_class(args.model_name)(args, metadata)
    Y = tmp_model(X)
    
    model = keras.models.Model(X, Y)
    model_name = tmp_model.model_name

    return model, model_name


# In[3]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--dataset', type=str, choices=['fc', 'kc'], default='fc')
    parser.add_argument('--max_category', type=int, default=0)
    parser.add_argument('--neighbor_threshold', type=float, default=999)
    parser.add_argument('--use_latlon', action='store_true')
    parser.add_argument('--use_waterfront', action='store_true')
    parser.add_argument('--model_name', type=str, default=f'MyBasic')
    parser.add_argument('--restore_model', action='store_true')
    parser.add_argument('--train_again', action='store_true')
    parser.add_argument('--D', type=int, default=128) # hidden dimension
    parser.add_argument('--L', type=int, default=3) # stack of layers
    parser.add_argument('--val_ratio', type=float, default=0.1)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default=f'adam')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience_stop', type=int, default=5)
    parser.add_argument('--patience_lr', type=int, default=2)
    
    args = parser.parse_args()
    
    if not os.path.isdir('prediction'):
        os.mkdir('prediction')
    if not os.path.isdir('test_logs'):
        os.mkdir('test_logs')
    if not os.path.isdir(f'prediction/{args.dataset}'):
        os.mkdir(f'prediction/{args.dataset}')
    if not os.path.isdir(f'test_logs/{args.dataset}'):
        os.mkdir(f'test_logs/{args.dataset}')
        
        
    dataset, metadata = utils.dataloader.load_data_ours(args)
    X_train, y_train, X_test, y_test = dataset
    print('X_train:', X_train.shape,   '\ty_train:', y_train.shape)
    print('X_test:', X_test.shape,    '\ty_test:', y_test.shape)
    
    
    print(args)
    slack_message += str(args) + '\n'
    
    
    model, model_name = model_define(args, metadata)
    model_logging_name = f'{args.dataset}_{model_name}_{args.neighbor_threshold}_{args.D}'
    model_checkpoint = f'./model_checkpoint/{args.dataset}/{model_logging_name}'
    model_logs = f'./model_logs/{args.dataset}/{model_logging_name}'
    
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate = args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = keras.optimizers.Adam(args.learning_rate)
    elif args.optimizer == 'adagrad':
        optimizer = keras.optimizers.Adagrad(args.learning_rate)

    model.compile(loss='mae', optimizer=optimizer)
    model.summary()

    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience_stop, min_delta=1e-6)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=args.patience_lr)
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(model_checkpoint, save_weights_only=True, \
                    save_best_only=True, monitor='val_loss', mode='min', verbose=0)
#     time_callback = utils.TimeHistory()
    # tb_callback = TensorBoard(log_dir=model_logs, histogram_freq=1, write_graph=True, write_images=True)
#     logging_callback = LoggingCallback()

    
    logging.basicConfig(filename=f'./test_logs/{args.dataset}/{model_logging_name}.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')
    logging.info(str(args))
    
    # Custom callback for logging metrics during training and testing
    class LoggingCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                logging.info(f"Epoch {epoch + 1}: "+ str(logs))
                
            if (epoch+1) % 5 == 0:
                predY = self.model.predict(X_test, batch_size=args.batch_size)
                y_pred = model.predict(X_test, batch_size=args.batch_size)
                print(f'Test epoch: {epoch+1}', args.dataset, utils.metric(np.exp(y_test), np.exp(y_pred)))
                logging.info(f'Test epoch: {epoch+1} \t {args.dataset} \t {utils.metric(np.exp(y_test), np.exp(y_pred))}')


    logging_callback = LoggingCallback()


    model.fit(X_train, y_train,
                batch_size=args.batch_size,
                epochs=args.max_epoch,
                verbose=1,
                validation_split=0.1,
                callbacks=[early_stopping, model_ckpt, reduce_lr, logging_callback],
    )

    model, model_name = model_define(args, metadata)
    model.load_weights(model_checkpoint)
    
    y_pred = model.predict(X_test, batch_size=args.batch_size)
    print(model_name, args.dataset, utils.metric(np.exp(y_test), np.exp(y_pred)))

    np.save(f'prediction/{args.dataset}/ground_truth.npy', y_test)
    np.save(f'prediction/{args.dataset}/{model_logging_name}.npy', y_pred)

