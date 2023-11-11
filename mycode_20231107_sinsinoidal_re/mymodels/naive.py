import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *

class MyNaiveEmb(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.model_name = f'MyNaiveEmb'
        self.num_features = metadata['num_features']
        self.num_neighbors = metadata['num_neighbors']
        self.categories = metadata['categories']
        self.X_ref = metadata['X_ref']
        self.y_ref = metadata['y_ref']
        self.D = args.D
        
    def build(self, input_shape):
        self.dense = Sequential([layers.Dense(self.D, activation='relu'),
                                            layers.Dense(self.D),])
        self.norm = layers.Normalization(-1)

    def call(self, X):
        batch_size = tf.shape(X)[0]
        
        X_emb = self.dense(X)
        return self.norm(X_emb)
    
    
    
class MyNaive(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.args = args
        self.metadata = metadata
        self.model_name = f'MyNaive'
        self.num_features = metadata['num_features']
        self.num_neighbors = metadata['num_neighbors']
        self.categories = metadata['categories']
        self.X_ref = metadata['X_ref']
        self.y_ref = metadata['y_ref']
        self.D = args.D
        
    def build(self, input_shape):
        self.naive_emb_layer = MyNaiveEmb(self.args, self.metadata)
        
        self.output_layer = Sequential([
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(1)])

    def call(self, X, S):
        batch_size = tf.shape(X)[0]
        
        X_emb = self.naive_emb_layer(X)
        
        output = self.output_layer(X_emb)
        return output
    
    
class MyNaiveNoise(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.args = args
        self.metadata = metadata
        self.model_name = f'MyNaiveNoise'
        self.num_features = metadata['num_features']
        self.num_neighbors = metadata['num_neighbors']
        self.categories = metadata['categories']
        self.X_ref = metadata['X_ref']
        self.y_ref = metadata['y_ref']
        self.D = args.D
        
    def build(self, input_shape):
        self.naive_emb_layer = MyNaiveEmb(self.args, self.metadata)
        
        self.output_layer = Sequential([
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(1)])

    def call(self, X, S):
        batch_size = tf.shape(X)[0]
        
        noise = tf.random.normal((batch_size, 2)) * .01
        noise_zero = tf.zeros((batch_size, X.shape[1]-2))
        noise = tf.concat((noise, noise_zero), -1)
        
        X = X + noise
        
        X_emb = self.naive_emb_layer(X)
        
        output = self.output_layer(X_emb)
        return output
    
    
    
# class MyBasicCat(tf.keras.layers.Layer):
#     def __init__(self, args, metadata):
#         super().__init__()
#         self.model_name = f'MyBasicCat'
#         self.num_features = metadata['num_features']
#         self.num_neighbors = metadata['num_neighbors']
#         self.categories = metadata['categories']
#         self.X_ref = metadata['X_ref']
#         self.y_ref = metadata['y_ref']
#         self.D = args.D
        
#     def build(self, input_shape):
#         self.dense_per_cat = [Sequential([layers.Dense(self.D, activation='relu'),
#                                             layers.Normalization(-1),
#                                             layers.Dense(self.D)]) for _ in range(len(self.categories))]
        
#         self.output_layer = Sequential([
#                                 layers.Dense(self.D, activation='relu'),
#                                 layers.Dense(self.D, activation='relu'),
#                                 layers.Dense(1)])

#     def call(self, X, S):
#         batch_size = tf.shape(X)[0]
        
#         X_emblist = []
#         for j in range(self.num_features):
#             Xj = X[..., j]
#             if self.categories[j] > 0:
#                 Xj = tf.one_hot(tf.cast(Xj, dtype=tf.int32), depth=self.categories[j])
#             else:
#                 Xj = tf.expand_dims(Xj, -1)
            
#             X_emb = self.dense_per_cat[j](Xj)
#             X_emblist.append(X_emb)
            
#         X_emb = tf.reduce_sum(X_emblist, 0)
        
#         output = self.output_layer(X_emb)
#         return output