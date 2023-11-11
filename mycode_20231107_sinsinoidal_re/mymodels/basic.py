import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *

    
# class MyBasic(tf.keras.layers.Layer):
#     def __init__(self, args, metadata):
#         super().__init__()
#         self.args = args
#         self.metadata = metadata
#         self.model_name = f'MyBasic'
#         self.num_features = metadata['num_features']
#         self.y_mean = metadata['y_mean']
#         self.y_std = metadata['y_std']
#         self.D = args.D
        
#     def build(self, input_shape):
#         self.output_layer = Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(1)])

#     def call(self, X):
#         batch_size = tf.shape(X)[0]
#         output = self.output_layer(X) 
#         return output * self.y_std + self.y_mean
    
    
# class MyBasicAddr(tf.keras.layers.Layer):
#     def __init__(self, args, metadata):
#         super().__init__()
#         self.args = args
#         self.metadata = metadata
#         self.model_name = f'MyBasicAddr'
#         self.num_features = metadata['num_features']
#         self.y_mean = metadata['y_mean']
#         self.y_std = metadata['y_std']
#         self.D = args.D
#         self.addr_nums = metadata['saddr_nums']
#         self.adj_mx = metadata['adj_mx']
        
#     def build(self, input_shape):
#         self.input_layer = Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(self.D),
#                                 layers.Normalization()
#                                 ])

#         self.addr_emb_1 = self.add_weight(shape=(self.addr_nums[0], self.D),
#                                             initializer='random_normal',
#                                             trainable=True, name='addr_emb_1')
#         #self.addr_emb_layer_norm_1 = layers.Normalization()

#         # self.addr_emb_layers = []
#         # for an in self.addr_nums:
#         #     self.addr_emb_layers.append(layers.Embedding(an, 32, embeddings_initializer='zeros'))
#         #self.addr_emb_layer_2 = layers.Embedding(self.addr_nums[1], 4, embeddings_initializer='zeros')
#         #self.addr_emb_layer_norm_2 = layers.Normalization()
                                

#         self.addr_layer = Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 #layers.Normalization()
#                                 ])
                                
#         self.output_layer = Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 # layers.Dropout(.2),
#                                 # layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(1)])

#     def call(self, X, Addr):
#         batch_size = tf.shape(X)[0]
#         X = self.input_layer(X)

#         addr_embs = []
#         # for i in range(len(self.addr_emb_layers)):
#         #     addr_emb.append(self.addr_emb_layers[i](Addr[:, i]))

#         #addr_emb_1 = self.addr_emb_layer_norm_1(self.addr_emb_1)
#         addr_emb_1 = self.addr_emb_1
#         addr_emb_1 = 0.8*(self.adj_mx @ addr_emb_1) + 0.2*addr_emb_1


#         addr_embs.append(tf.gather(addr_emb_1, Addr[:, 0]))
#         # addr_embs.append(self.addr_emb_layer_norm_2(self.addr_emb_layer_2(Addr[:, 1])))
#         # addr_embs.append(self.addr_emb_layer_2(Addr[:, 1]))
#         addr_emb = tf.concat(addr_embs, -1)
#         addr_emb = self.addr_layer(addr_emb)

#         # output = self.output_layer(X + addr_emb)
#         output = self.output_layer(tf.concat((X, addr_emb), -1))


#         return output * self.y_std + self.y_mean
    
    
# class MyBasicAddrNei(tf.keras.layers.Layer):
#     def __init__(self, args, metadata):
#         super().__init__()
#         self.args = args
#         self.metadata = metadata
#         self.model_name = f'MyBasicAddrNei'
#         self.num_features = metadata['num_features']
#         self.y_mean = metadata['y_mean']
#         self.y_std = metadata['y_std']
#         self.D = args.D
#         self.K = args.K
#         self.d = args.d
#         self.addr_nums = metadata['saddr_nums']
#         self.adj_mx = metadata['adj_mx']
#         self.max_neighboridx = metadata['max_neighboridx']
#         self.train_features = metadata['Train_features']
        
#     def build(self, input_shape):
#         self.input_layer = Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(self.D),
#                                 layers.Normalization()
#                                 ])

#         # self.house_emb = self.add_weight(shape=(self.max_neighboridx, self.D),
#         #                                     initializer='random_normal',
#         #                                     trainable=True, name='house_emb')
#         self.house_emb_layer_q = Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(self.D),
#                                 layers.Normalization()
#                                 ])
#         self.house_emb_layer_k = Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(self.D),
#                                 layers.Normalization()
#                                 ])
#         self.house_emb_layer_v = Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(self.D),
#                                 layers.Normalization()
#                                 ])
#         self.house_emb_out = Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(self.D),
#                                 layers.Normalization()
#                                 ])

#         # self.addr_emb_1 = self.add_weight(shape=(self.addr_nums[0], self.D),
#         #                                     initializer='random_normal',
#         #                                     trainable=True, name='addr_emb_1')

#         # self.addr_layer = Sequential([
#         #                         layers.Dense(self.D, activation='leaky_relu'),
#         #                         layers.Normalization()
#         #                         ])
                                
#         self.output_layer = Sequential([
#                                 layers.Dense(self.D, activation='leaky_relu'),
#                                 # layers.Dropout(.2),
#                                 # layers.Dense(self.D, activation='leaky_relu'),
#                                 layers.Dense(1)])

#     def call(self, X, Addr, Nidx, Ndist):
#         batch_size = tf.shape(X)[0]
#         X = self.input_layer(X)

#         addr_embs = []
#         # for i in range(len(self.addr_emb_layers)):
#         #     addr_emb.append(self.addr_emb_layers[i](Addr[:, i]))

#         #addr_emb_1 = self.addr_emb_layer_norm_1(self.addr_emb_1)
#         # addr_emb_1 = self.addr_emb_1
#         # addr_emb_1 = 0.8*(self.adj_mx @ addr_emb_1) + 0.2*addr_emb_1


#         # addr_embs.append(tf.gather(addr_emb_1, Addr[:, 0]))
#         # # addr_embs.append(self.addr_emb_layer_norm_2(self.addr_emb_layer_2(Addr[:, 1])))
#         # # addr_embs.append(self.addr_emb_layer_2(Addr[:, 1]))
#         # addr_emb = tf.concat(addr_embs, -1)
#         # addr_emb = self.addr_layer(addr_emb)


        

#         q_emb = self.house_emb_layer_q(X)  # (batch, D)
#         q_emb = tf.expand_dims(q_emb, 1)   # (batch, 1, D)
#         k_emb = self.house_emb_layer_k(tf.gather(self.train_features, Nidx)) # (batch, 60, D)
#         v_emb = self.house_emb_layer_v(tf.gather(self.train_features, Nidx)) # (batch, 60, D)


#         q_emb = tf.concat(tf.split(q_emb, self.K, axis = -1), axis = 0)
#         k_emb = tf.concat(tf.split(k_emb, self.K, axis = -1), axis = 0)
#         v_emb = tf.concat(tf.split(v_emb, self.K, axis = -1), axis = 0)

#         # query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
#         # key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
#         # value = tf.concat(tf.split(value, K, axis = -1), axis = 0)

#         k_emb = tf.transpose(k_emb, perm = (0, 2, 1))
#         attention = tf.matmul(q_emb, k_emb)
#         attention /= (self.d ** 0.5)
#         attention = tf.nn.softmax(attention, axis = -1)
#         neigh_emb = tf.matmul(attention, v_emb)
#         neigh_emb = tf.squeeze(neigh_emb, 1)

#         neigh_emb = tf.concat(tf.split(neigh_emb, self.K, axis = 0), axis = -1)
#         neigh_emb = self.house_emb_out(neigh_emb)
#         # neigh_emb = self.house_emb_layer(tf.gather(self.train_features, Nidx))
#         # neigh_emb = tf.reduce_mean(neigh_emb, -2)


#         # output = self.output_layer(X + addr_emb)
#         # output = self.output_layer(tf.concat((X, addr_emb, neigh_emb), -1))
#         output = self.output_layer(tf.concat((X, neigh_emb), -1))


#         return output * self.y_std + self.y_mean
    
import tensorflow as tf
from positional_encodings.tf_encodings import TFPositionalEncoding2D, TFSummer





class MyBasicNei(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.args = args
        self.metadata = metadata
        self.model_name = f'MyBasicNei'
        self.num_features = metadata['num_features']
        self.y_mean = metadata['y_mean']
        self.y_std = metadata['y_std']
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.sigma = args.sigma
        #self.adj_mx = metadata['adj_mx']
        self.max_neighboridx = metadata['max_neighboridx']
        self.train_features = metadata['Train_features']
        self.ncols = metadata['ncols']
        self.nrows = metadata['nrows']
        self.args = args

        self.grid_emb = metadata['grid_emb']
        if args.use_sinsinoidal:
            p_enc_2d = TFPositionalEncoding2D(64)
            y = tf.zeros((1,100,100,64))
            image = p_enc_2d(y).numpy()
            image = image.squeeze(0)
            self.grid_emb = image


        
    def build(self, input_shape):
        self.input_layer = Sequential([
                                layers.Dense(self.D, activation='elu'),
                                layers.Dense(self.D),
                                ])
                                
        # self.grid_emb = self.add_weight(shape=(1, self.nrows, self.ncols, self.D),
        #                                     initializer='random_normal',
        #                                     trainable=True, name='grid_emb')
        #self.grid_emb = positional_encoding_2d((self.nrows, self.ncols), self.D)

        #self.grid_cnn = tf.keras.layers.Conv2D(self.D, 3, padding="same")

        # self.house_emb = self.add_weight(shape=(self.max_neighboridx, self.D),
        #                                     initializer='random_normal',
        #                                     trainable=True, name='house_emb')
        # self.house_emb_layer = Sequential([
        #                         layers.Dense(self.D, activation='relu'),
        #                         layers.Dense(self.D),
        #                         layers.Normalization()
        #                         ])
        self.house_emb_layer_q = Sequential([
                                layers.Dense(self.D, activation='elu'),
                                layers.Dense(self.D),
                                ])
        self.house_emb_layer_k = Sequential([
                                layers.Dense(self.D, activation='elu'),
                                layers.Dense(self.D),
                                ])
        self.house_emb_layer_v = Sequential([
                                layers.Dense(self.D, activation='elu'),
                                layers.Dense(self.D),
                                ])
                                
        self.house_emb_layer2_q = Sequential([
                                layers.Dense(self.D, activation='elu'),
                                layers.Dense(self.D),
                                ])
        self.house_emb_layer2_k = Sequential([
                                layers.Dense(self.D, activation='elu'),
                                layers.Dense(self.D),
                                ])
        self.house_emb_layer2_v = Sequential([
                                layers.Dense(self.D, activation='elu'),
                                layers.Dense(self.D),
                                ])
        # self.house_emb_out = Sequential([
        #                         layers.Dense(self.D, activation='elu'),
        #                         layers.Dense(self.D),
        #                         ])


        self.output_layer = Sequential([
                                # layers.Dense(self.D, activation='elu'),
                                layers.Dense(self.D, activation='elu'),
                                # layers.Dense(self.D, activation='leaky_relu'),
                                layers.Dense(1)])

    def call(self, X, Nidx, Ndist, Eidx, Edist):
        batch_size = tf.shape(X)[0]

        X_ij = X[:, :2]
        X_idx = tf.cast(X_ij[:, 0] + X_ij[:, 1] * self.ncols, tf.int32)

        X = self.input_layer(X[:, 2:])

        grid_emb = tf.expand_dims(self.grid_emb, 0)
        # grid_emb = self.grid_cnn(grid_emb)


        grid_emb = tf.squeeze(grid_emb, 0)
        grid_emb = tf.reshape(grid_emb, (grid_emb.shape[0]*grid_emb.shape[1], -1))
        grid_emb = tf.gather(grid_emb, X_idx)



        ################
        q_emb = self.house_emb_layer_q(X)  # (batch, D)
        q_emb = tf.expand_dims(q_emb, 1)   # (batch, 1, D)
        k_emb = self.house_emb_layer_k(tf.gather(self.train_features, Nidx)) # (batch, 60, D)
        v_emb = self.house_emb_layer_v(tf.gather(self.train_features, Nidx)) # (batch, 60, D)

        q_emb = tf.concat(tf.split(q_emb, self.K, axis = -1), axis = 0)
        k_emb = tf.concat(tf.split(k_emb, self.K, axis = -1), axis = 0)
        v_emb = tf.concat(tf.split(v_emb, self.K, axis = -1), axis = 0)

        k_emb = tf.transpose(k_emb, perm = (0, 2, 1))
        attention = tf.matmul(q_emb, k_emb)
        attention /= (self.d ** 0.5)

        mask = Ndist < self.sigma
        mask = tf.expand_dims(mask, 1)
        mask = tf.tile(mask, multiples = (self.K, 1, 1))

        attention = tf.compat.v2.where(
            condition = mask, x = attention, y = -2 ** 15 + 1)

        # masking
        attention = tf.nn.softmax(attention, axis = -1)
        neigh_emb = tf.matmul(attention, v_emb)
        neigh_emb = tf.squeeze(neigh_emb, 1)

        neigh_emb = tf.concat(tf.split(neigh_emb, self.K, axis = 0), axis = -1)


        
        ################
        q_emb = self.house_emb_layer2_q(X)  # (batch, D)
        q_emb = tf.expand_dims(q_emb, 1)   # (batch, 1, D)
        k_emb = self.house_emb_layer2_k(tf.gather(self.train_features, Eidx)) # (batch, 60, D)
        v_emb = self.house_emb_layer2_v(tf.gather(self.train_features, Eidx)) # (batch, 60, D)

        q_emb = tf.concat(tf.split(q_emb, self.K, axis = -1), axis = 0)
        k_emb = tf.concat(tf.split(k_emb, self.K, axis = -1), axis = 0)
        v_emb = tf.concat(tf.split(v_emb, self.K, axis = -1), axis = 0)

        k_emb = tf.transpose(k_emb, perm = (0, 2, 1))
        attention = tf.matmul(q_emb, k_emb)
        attention /= (self.d ** 0.5)

        mask = Edist < 0.02
        mask = tf.expand_dims(mask, 1)
        mask = tf.tile(mask, multiples = (self.K, 1, 1))

        attention = tf.compat.v2.where(
            condition = mask, x = attention, y = -2 ** 15 + 1)

        # masking
        attention = tf.nn.softmax(attention, axis = -1)
        neigh_emb2 = tf.matmul(attention, v_emb)
        neigh_emb2 = tf.squeeze(neigh_emb2, 1)

        neigh_emb2 = tf.concat(tf.split(neigh_emb2, self.K, axis = 0), axis = -1)



        if self.args.use_latlon:
            output = self.output_layer(tf.concat((X, neigh_emb, neigh_emb2, grid_emb), -1))
        else:
            output = self.output_layer(tf.concat((X, neigh_emb, neigh_emb2), -1))
        # output = self.output_layer(X + neigh_emb)


        return output * self.y_std + self.y_mean
    

    