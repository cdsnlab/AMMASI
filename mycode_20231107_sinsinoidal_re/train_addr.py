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
import slack_test


slack_message = ""

from datetime import datetime 

start_time = datetime.now() 


# In[2]:


def model_define(args, metadata):
    num_features = metadata['num_features']
    #num_addrs = len(metadata['saddr_nums'])
    num_neighbors = metadata['num_neighbors']

    X = layers.Input(shape=(num_features,), dtype=tf.float32) # X_feature
    #Addr = layers.Input(shape=(num_addrs,), dtype=tf.int32) # X_feature
    Nidx = layers.Input(shape=(num_neighbors,), dtype=tf.int32) # X_feature
    Ndist = layers.Input(shape=(num_neighbors,), dtype=tf.float32) # X_feature
    Eidx = layers.Input(shape=(num_neighbors,), dtype=tf.int32) # X_feature
    Edist = layers.Input(shape=(num_neighbors,), dtype=tf.float32) # X_feature
    
    tmp_model = mymodels.str_to_class(args.model_name)(args, metadata)
    Y = tmp_model(X, Nidx, Ndist, Eidx, Edist)
    
    model = keras.models.Model((X, Nidx, Ndist, Eidx, Edist), Y)
    model_name = tmp_model.model_name

    return model, model_name


# In[3]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameter')
    parser.add_argument('--dataset', type=str, choices=['fc', 'kc', 'sp', 'poa'], default='fc')
    parser.add_argument('--use_latlon', action='store_true')
    parser.add_argument('--use_locfeat', action='store_true')

    parser.add_argument('--use_sinsinoidal', action='store_true')
    parser.add_argument('--use_poiprox', action='store_true')
    parser.add_argument('--model_name', type=str, default=f'MyBasicNei')
    parser.add_argument('--restore_model', action='store_true')
    parser.add_argument('--train_again', action='store_true')
    parser.add_argument('--D', type=int, default=64) # hidden dimension
    parser.add_argument('--K', type=int, default=8) # stack of layers
    parser.add_argument('--d', type=int, default=8) # stack of layers
    parser.add_argument('--sigma', type=float, default=0.02) # stack of layers
    parser.add_argument('--sigma2', type=float, default=0.02) # stack of layers
    parser.add_argument('--val_ratio', type=float, default=0.1)
    
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default=f'adam')
    parser.add_argument('--learning_rate', type=float, default=0.008)
    parser.add_argument('--patience_stop', type=int, default=10)
    parser.add_argument('--patience_lr', type=int, default=5)
    
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
    #X_train, Addr_train, y_train, X_test, Addr_test, y_test = dataset
    
    # X_train, Addr_train, Nidx_train, Ndist_train, y_train, \
    #     X_test , Addr_test , Nidx_test , Ndist_test , y_test = dataset
    # X_train, Nidx_train, Ndist_train, y_train, \
    #     X_test , Nidx_test , Ndist_test , y_test = dataset
    X_train, Nidx_train, Ndist_train, Eidx_train, Edist_train, y_train, \
            X_test , Nidx_test , Ndist_test , Eidx_test, Edist_test, y_test = dataset
    print('X_train:', X_train.shape,  'y_train:', y_train.shape, sep='\t')
    print('X_test:', X_test.shape, 'y_test:', y_test.shape, sep='\t')
    
    
    print(args)
    slack_message += str(args) + '\n'
    
    
    model, model_name = model_define(args, metadata)
    latlon_type = args.use_latlon
    if args.use_latlon and args.use_sinsinoidal:
        latlon_type = 'Sinsinoidal'
    if args.use_locfeat:
        latlon_type = 'Locfeat'
    if args.use_locfeat and args.use_latlon:
        import sys
        sys.exit(-1)
    model_logging_name = f'{args.dataset}_{model_name}_{args.D}_{args.sigma}_{args.sigma2}_loc_{latlon_type}_poi_{args.use_poiprox}'
    model_checkpoint = f'./model_checkpoint/{args.dataset}/{model_logging_name}'
    model_logs = f'./model_logs/{args.dataset}/{model_logging_name}'


    # if os.path.isfile(f'prediction/{args.dataset}/{model_logging_name}.npy') and not args.use_latlon:
    #     import sys
    #     sys.exit(0)
    
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate = args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = keras.optimizers.Adam(args.learning_rate)
    elif args.optimizer == 'adagrad':
        optimizer = keras.optimizers.Adagrad(args.learning_rate)

    from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
    model.compile(loss='mae', optimizer=optimizer, metrics=[RootMeanSquaredError()])
    model.summary()

    # Define some callbacks to improve training.            
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience_stop)
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
                y_pred = self.model.predict((X_test, Nidx_test, Ndist_test, Eidx_test, Edist_test), batch_size=args.batch_size, verbose=0)
                print('TESTING...', f'Test epoch: {epoch+1}', args.dataset, utils.metric(np.exp(y_test), np.exp(y_pred)))
                print(f'Test epoch: {epoch+1}', args.dataset, utils.metric(np.exp(y_test), np.exp(y_pred)))
                logging.info(f'Test epoch: {epoch+1} \t {args.dataset} \t {utils.metric(np.exp(y_test), np.exp(y_pred))}')


    logging_callback = LoggingCallback()


    model.fit((X_train, Nidx_train, Ndist_train, Eidx_train, Edist_train), y_train,
                batch_size=args.batch_size,
                epochs=args.max_epoch,
                verbose=1,
                validation_split=0.1,
                callbacks=[early_stopping, model_ckpt, reduce_lr, logging_callback],
    )

    model, model_name = model_define(args, metadata)
    model.load_weights(model_checkpoint)
    
    y_pred = model.predict((X_test, Nidx_test, Ndist_test, Eidx_test, Edist_test), batch_size=args.batch_size)

    print(model_name, args.dataset, utils.metric(np.exp(y_test), np.exp(y_pred)))


    #########################
    slack_message += str(f'{model_logging_name} result') + '\n'
    slack_message += f'{model_name}, {args.dataset}, {utils.metric(np.exp(y_test), np.exp(y_pred))}\n'
    time_elapsed = datetime.now() - start_time 
    slack_message += f'Time elapsed (hh:mm:ss ms) {time_elapsed}\n'
    slack_message += f'------------------------------------------\n'
    slack_test.slackBot.print(slack_message)


    np.save(f'prediction/{args.dataset}/ground_truth.npy', y_test)
    np.save(f'prediction/{args.dataset}/{model_logging_name}.npy', y_pred)

