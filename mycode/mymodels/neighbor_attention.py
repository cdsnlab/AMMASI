import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from mymodels.basic import *
from mymodels.naive import *



class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)
   
    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x



class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x




class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x



class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x



class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x


class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    
    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.


    
    
class MyBasicNeighborAtt(tf.keras.layers.Layer):
    def __init__(self, args, metadata):
        super().__init__()
        self.args = args
        self.metadata = metadata
        self.model_name = f'MyBasicNeighborAtt'
        self.num_features = metadata['num_features']
        self.num_neighbors = metadata['num_neighbors']
        self.categories = metadata['categories']
        self.X_ref = tf.convert_to_tensor(metadata['X_ref'], dtype=tf.float32)
        self.y_ref = tf.convert_to_tensor(metadata['y_ref'], dtype=tf.float32)
        self.D = args.D
        self.L = args.L
        
    def build(self, input_shape):
        self.basic_emb_layer = MyBasicEmb(self.args, self.metadata)
        self.neighbor_basic_emb_layer = MyBasicYEmb(self.args, self.metadata)
        
        self.output_layer = Sequential([
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(self.D, activation='relu'),
                                layers.Dense(1)])
        
        #self.cross_att = CrossAttention(num_heads=8, key_dim=self.D)
        self.mha = [tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=self.D) for _ in range(self.L)]
        self.layernorm = [tf.keras.layers.LayerNormalization() for _ in range(self.L)]
        self.add = [tf.keras.layers.Add() for _ in range(self.L)]

    def call(self, X, S):
        batch_size = tf.shape(X)[0]
        
        X_emb = self.basic_emb_layer(X)
        
        #S_count = tf.reduce_sum(tf.cast(S > 0, dtype=tf.float32), -1) + 1
        #S_count = tf.expand_dims(S_count, 1)
        X_near = tf.gather(self.X_ref, S, axis=0)
        y_near = tf.expand_dims(tf.gather(self.y_ref, S, axis=0), -1)
        
        X_near = tf.concat((X_near, y_near), -1)
        X_near = tf.reshape(X_near, (-1, X_near.shape[-1]))
        X_near = self.neighbor_basic_emb_layer(X_near)
        X_near = tf.reshape(X_near, (-1, self.num_neighbors, X_near.shape[-1]))
        #X_near = tf.reduce_sum(X_near, 1) / S_count
        
        
        attention_mask = tf.expand_dims(tf.cast(S >= 0, dtype=tf.int32), 1)
        X_emb = tf.expand_dims(X_emb, 1) 
        for i in range(self.L):
            query = X_emb
            x = query
            context = X_near

            attn_output, attn_scores = self.mha[i](
                query=query,
                key=context,
                value=context,
                attention_mask=attention_mask,
                return_attention_scores=True)

            # Cache the attention scores for plotting later.
            self.last_attn_scores = attn_scores
            x = self.add[i]([x, attn_output])
            x = self.layernorm[i](x)

            X_emb = x
            
        X_emb = tf.squeeze(X_emb, 1)
        
        #X_emb = tf.concat((X_emb, X_near), -1)
        output = self.output_layer(X_emb)
        return output