import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from mymodels.basic import *

    
class MyNeighborMean(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.args = args
        self.metadata = metadata
        self.model_name = f'MyNeighborMean'
        self.num_features = metadata['num_features']
        self.num_neighbors = metadata['num_neighbors']
        self.categories = metadata['categories']
        self.X_ref = tf.convert_to_tensor(metadata['X_ref'], dtype=tf.float32)
        self.y_ref = tf.convert_to_tensor(metadata['y_ref'], dtype=tf.float32)
        self.D = args.D
        
    def build(self, input_shape):
        self.basic_emb_layer = MyBasicEmb(self.args, self.metadata)
        self.neighbor_basic_emb_layer = MyBasicYEmb(self.args, self.metadata)
        
        self.output_layer = Sequential([
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(1)])

    def call(self, X, S):
        batch_size = tf.shape(X)[0]
        
        X_emb = self.basic_emb_layer(X)
        
        S_count = tf.reduce_sum(tf.cast(S >= 0, dtype=tf.float32), -1) + 1
        S_count = tf.expand_dims(S_count, 1)
        y_near = tf.expand_dims(tf.gather(self.y_ref, S, axis=0), -1)
        y_near = tf.reduce_sum(y_near, 1) / S_count
        
        X_emb = tf.concat((X_emb, y_near), -1) #tf.concat((X_emb, X_near), -1)
        output = self.output_layer(X_emb)
        return output
    
    
# class MyBasicNeighbor(tf.keras.layers.Layer):
#     def __init__(self, args, metadata):
#         super().__init__()
#         self.args = args
#         self.metadata = metadata
#         self.model_name = f'MyBasicNeighbor'
#         self.num_features = metadata['num_features']
#         self.num_neighbors = metadata['num_neighbors']
#         self.categories = metadata['categories']
#         self.X_ref = tf.convert_to_tensor(metadata['X_ref'], dtype=tf.float32)
#         self.y_ref = tf.convert_to_tensor(metadata['y_ref'], dtype=tf.float32)
#         self.D = args.D
        
#     def build(self, input_shape):
#         self.basic_emb_layer = MyBasicEmb(self.args, self.metadata)
#         self.neighbor_basic_emb_layer = MyBasicEmb(self.args, self.metadata)
        
#         self.output_layer = Sequential([
#                                 layers.Dense(self.D, activation='relu'),
#                                 layers.Dense(self.D, activation='relu'),
#                                 layers.Dense(1)])

#     def call(self, X, S):
#         batch_size = tf.shape(X)[0]
        
#         X_emb = self.basic_emb_layer(X, S)
#         y_nearsum = tf.reduce_sum((tf.gather(self.y_ref, S, axis=0) * tf.cast(S > 0, dtype=tf.float32)), -1)
#         S_count = tf.reduce_sum(tf.cast(S > 0, dtype=tf.float32), -1)+1
#         X_emb = tf.concat((X_emb, (y_nearsum/S_count)[:, tf.newaxis]), -1)
        
        
        
#         output = self.output_layer(X_emb)
#         return output
    
    
class MyBasicNeighbor(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.args = args
        self.metadata = metadata
        self.model_name = f'MyBasicNeighbor'
        self.num_features = metadata['num_features']
        self.num_neighbors = metadata['num_neighbors']
        self.categories = metadata['categories']
        self.X_ref = tf.convert_to_tensor(metadata['X_ref'], dtype=tf.float32)
        self.y_ref = tf.convert_to_tensor(metadata['y_ref'], dtype=tf.float32)
        self.D = args.D
        
    def build(self, input_shape):
        self.basic_emb_layer = MyBasicEmb(self.args, self.metadata)
        self.neighbor_basic_emb_layer = MyBasicYEmb(self.args, self.metadata)
        
        self.output_layer = Sequential([
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(1)])

    def call(self, X, S):
        batch_size = tf.shape(X)[0]
        
        X_emb = self.basic_emb_layer(X)
        
        S_count = tf.reduce_sum(tf.cast(S >= 0, dtype=tf.float32), -1) + 1
        S_count = tf.expand_dims(S_count, 1)
        X_near = tf.gather(self.X_ref, S, axis=0)
        y_near = tf.expand_dims(tf.gather(self.y_ref, S, axis=0), -1)
        X_near = tf.concat((X_near, y_near), -1)
        X_near = tf.reshape(X_near, (-1, X_near.shape[-1]))
        X_near = self.neighbor_basic_emb_layer(X_near)
        X_near = tf.reshape(X_near, (-1, self.num_neighbors, X_near.shape[-1]))
        X_near = tf.reduce_sum(X_near, 1) / S_count
        
        
        X_emb = (X_emb + X_near) / 2 #tf.concat((X_emb, X_near), -1)
        output = self.output_layer(X_emb)
        return output
    
    