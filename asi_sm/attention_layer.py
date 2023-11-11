from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

import tensorflow as tf
from tensorflow.keras.layers import multiply, RepeatVector
from tensorflow.keras.layers import Lambda, Permute
from asi_sm.distance import Distance
from asi_sm.transformation import CompFunction


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)

class NeighborAttention(BaseAttention):
  def call(self, x, y):
    attn_output = self.mha(
        query=x,
        value=y)
    return attn_output

class Attention(Layer):

    def __init__(self,
                 sigma,
                 num_nearest,
                 shape_input_phenomenon,
                 type_compatibility_function,
                 num_features_extras,
                 calculate_distance=False,
                 graph_label=None,
                 phenomenon_structure_repeat=None,
                 context_structure=None,
                 type_distance=None,
                 suffix_mean=None,
                 **kwargs):

        self.sigma = sigma,
        self.num_nearest = num_nearest,
        self.shape_input_phenomenon = shape_input_phenomenon,
        self.type_compatibility_function = type_compatibility_function,
        self.num_features_extras = num_features_extras,
        self.calculate_distance = calculate_distance,
        self.graph_label = graph_label,
        self.phenomenon_structure_repeat = phenomenon_structure_repeat,
        self.context_structure = context_structure,
        self.type_distance = type_distance,
        self.suffix_mean = suffix_mean

        self.output_dim = self.shape_input_phenomenon[0] + self.num_features_extras[0]

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        # Initialize weights for each attention head
        # Layer kernel
        self.kernel = self.add_weight(shape=(self.num_nearest[0], self.num_nearest[0]),
                                 name='kernel_{}'.format(self.graph_label[0]))

        #Layer bias
        self.bias = self.add_weight(shape=(self.num_nearest[0],),
                               name='bias_{}'.format(self.graph_label[0]))

        self.built = True

    def call(self, inputs):
        source_distance = inputs[0]  # Node features (N x F)
        context = inputs[1]

        ######################## Attention data ########################

        if self.calculate_distance[0]:

            dist = Distance(self.phenomenon_structure_repeat[0], self.context_structure[0], self.type_distance[0])
            distance = dist.run()

        else:
            distance = source_distance

        # calculate the similarity measure of each neighbor (m, seq)
        comp_func = CompFunction(self.sigma[0], distance, self.type_compatibility_function[0], self.graph_label[0])
        simi = comp_func.run()

        # calculates the weights associated with each neighbor (m, seq)
        weight = K.dot(simi, self.kernel)
        weight = K.bias_add(weight, self.bias)
        weight = K.softmax(weight)

        # repeats the previous vector as many times as the feature number plus the point target and features extras
        # (input_phenomenon + 1 + num_features_extras, seq)
        prob_repeat = RepeatVector(self.shape_input_phenomenon[0] + self.num_features_extras[0])(weight)

        # inverts the dimensions in such a way that in each line,
        # we have the weight assigned to the neighbor (seq, input_phenomenon + 1 + num_features_extras)
        prob_repeat = Permute((2, 1))(prob_repeat)

        # multiplies each neighbor's feature by its respective weight
        # (seq, input_phenomenon + 1 + num_features_extras) x (seq, input_phenomenon + 1 + num_features_extras)
        relevance = multiply([prob_repeat, context])

        # add each column to find the mean vector (input_phenomenon + 1 + num_features_extras,)
        mean = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), name=self.suffix_mean[0])(relevance)

        return mean

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape



class AttentionGeo(Layer):

    def __init__(self,
                 sigma,
                 num_nearest,
                 shape_input_phenomenon,
                 type_compatibility_function,
                 num_features_extras,
                 calculate_distance=False,
                 graph_label=None,
                 phenomenon_structure_repeat=None,
                 context_structure=None,
                 type_distance=None,
                 suffix_mean=None,
                 **kwargs):

        self.sigma = sigma,
        self.num_nearest = num_nearest,
        self.shape_input_phenomenon = shape_input_phenomenon,
        self.type_compatibility_function = type_compatibility_function,
        self.num_features_extras = num_features_extras,
        self.calculate_distance = calculate_distance,
        self.graph_label = graph_label,
        self.phenomenon_structure_repeat = phenomenon_structure_repeat,
        self.context_structure = context_structure,
        self.type_distance = type_distance,
        self.suffix_mean = suffix_mean

        self.output_dim = self.shape_input_phenomenon[0] + self.num_features_extras[0]

        super(AttentionGeo, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        # Initialize weights for each attention head
        # Layer kernel
        self.kernel = self.add_weight(shape=(self.num_nearest[0], self.num_nearest[0]),
                                 name='kernel_{}'.format(self.graph_label[0]))

        #Layer bias
        self.bias = self.add_weight(shape=(self.num_nearest[0],),
                               name='bias_{}'.format(self.graph_label[0]))

        self.dense1_node2vec = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                                            tf.keras.layers.Dense(64),
                                                            tf.keras.layers.Normalization(-1)])
        self.dense2_node2vec = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                                            tf.keras.layers.Dense(64),
                                                            tf.keras.layers.Normalization(-1)])

        self.built = True

    def call(self, inputs):
        source_distance = inputs[0]  # Node features (N x F)
        context = inputs[1]
        node2vec = inputs[2]

        node2vec1 = self.dense1_node2vec(node2vec)
        node2vec2 = self.dense2_node2vec(node2vec)

        node2vec_target = node2vec1[:, :1, :]
        node2vec_neighbor = node2vec2[:, 1:, :]
        node2vec_target = tf.tile(node2vec_target, (1, node2vec_neighbor.shape[1], 1))

        ######################## Attention data ########################

        # if self.calculate_distance[0]:

        #     dist = Distance(self.phenomenon_structure_repeat[0], self.context_structure[0], self.type_distance[0])
        #     distance = dist.run()

        # else:
        distance = source_distance

        # calculate the similarity measure of each neighbor (m, seq)
        comp_func = CompFunction(self.sigma[0], distance, self.type_compatibility_function[0], self.graph_label[0])
        simi1 = comp_func.run()
        

        node2vec_target = K.l2_normalize(node2vec_target, axis=-1)
        node2vec_neighbor = K.l2_normalize(node2vec_neighbor, axis=-1)
        simi2 = tf.reduce_mean(node2vec_target * node2vec_neighbor, -1)

        # simi = simi1 #+ simi2*.5
        simi = simi1 + simi2*.1
        

        # calculates the weights associated with each neighbor (m, seq)
        weight = K.dot(simi, self.kernel)
        weight = K.bias_add(weight, self.bias)
        weight = K.softmax(weight)

        # repeats the previous vector as many times as the feature number plus the point target and features extras
        # (input_phenomenon + 1 + num_features_extras, seq)
        prob_repeat = RepeatVector(self.shape_input_phenomenon[0] + self.num_features_extras[0])(weight)

        # inverts the dimensions in such a way that in each line,
        # we have the weight assigned to the neighbor (seq, input_phenomenon + 1 + num_features_extras)
        prob_repeat = Permute((2, 1))(prob_repeat)

        # multiplies each neighbor's feature by its respective weight
        # (seq, input_phenomenon + 1 + num_features_extras) x (seq, input_phenomenon + 1 + num_features_extras)
        relevance = multiply([prob_repeat, context])

        # add each column to find the mean vector (input_phenomenon + 1 + num_features_extras,)
        mean = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), name=self.suffix_mean[0])(relevance)

        return mean

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape



class AttentionNew(Layer):

    def __init__(self,
                 #num_neuron, 
                 sigma,
                 num_nearest,
                 shape_input_phenomenon,
                 type_compatibility_function,
                 num_features_extras,
                 calculate_distance=False,
                 graph_label=None,
                 phenomenon_structure_repeat=None,
                 context_structure=None,
                 type_distance=None,
                 suffix_mean=None,
                 **kwargs):
        
        #self.num_neuron = num_neuron
        self.sigma = sigma,
        self.num_nearest = num_nearest,
        self.shape_input_phenomenon = shape_input_phenomenon,
        self.type_compatibility_function = type_compatibility_function,
        self.num_features_extras = num_features_extras,
        self.calculate_distance = calculate_distance,
        self.graph_label = graph_label,
        self.phenomenon_structure_repeat = phenomenon_structure_repeat,
        self.context_structure = context_structure,
        self.type_distance = type_distance,
        self.suffix_mean = suffix_mean

        self.output_dim = self.shape_input_phenomenon[0] + self.num_features_extras[0]

        super(AttentionNew, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        # Initialize weights for each attention head
        # Layer kernel
        #self.kernel = self.add_weight(shape=(self.num_nearest[0], self.num_nearest[0]),
        #                         name='kernel_{}'.format(self.graph_label[0]))

        #Layer bias
        #self.bias = self.add_weight(shape=(self.num_nearest[0],),
        #                       name='bias_{}'.format(self.graph_label[0]))
        self.neighbor_attention = NeighborAttention(
                                    num_heads=8,
                                    key_dim=64)
        # self.dense_n2v = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation='relu'),
        #                                                 tf.keras.layers.Dense(32),
        #                                                 tf.keras.layers.Normalization(-1)])
        # self.dense_context = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
        #                                                 tf.keras.layers.Dense(32)])
        # self.dense_phenomenon = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
        #                                                 tf.keras.layers.Dense(32)])
        self.dense_query = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                                        tf.keras.layers.Dense(64),
                                                        tf.keras.layers.Normalization(-1)])
        self.dense_value = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                                        tf.keras.layers.Dense(64),
                                                        tf.keras.layers.Normalization(-1)])
        self.dense_output = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                                        tf.keras.layers.Dense(64)])

        self.built = True

    def call(self, inputs):

        input_phenomenon = inputs[0]
        source_distance = inputs[1]  # Node features (N x F)
        print('self.input_phenomenon.shape', input_phenomenon)
        print('self.source_distance.shape', source_distance.shape)
        print('self.context.shape', inputs[2].shape)
        print('self.n2v.shape', inputs[3].shape)
        # phenomenon = tf.expand_dims(self.dense_phenomenon(input_phenomenon), 1)
        # context = self.dense_context(inputs[2])
        phenomenon = tf.expand_dims(input_phenomenon, 1)
        context = inputs[2]
        node2vec = inputs[3] #self.dense_n2v(inputs[3])


        # att_query = self.dense_query(tf.concat((node2vec[:, :1, :], phenomenon), -1))
        # att_value = self.dense_value(tf.concat((node2vec[:, 1:, :], context), -1))

        # att_query = node2vec[:, :1, :]
        # att_value = node2vec[:, 1:, :]

        
        att_query = self.dense_query(phenomenon)
        att_value = self.dense_value(context)

        att_output = self.neighbor_attention(att_query, att_value)[:, 0, :]
        att_output = self.dense_output(att_output)

        print('att_query.shape, att_value.shape', att_query.shape, att_value.shape)
        print('att_output', att_output.shape)

        return att_output
        

        # ######################## Attention data ########################

        # if self.calculate_distance[0]:

        #     dist = Distance(self.phenomenon_structure_repeat[0], self.context_structure[0], self.type_distance[0])
        #     distance = dist.run()

        # else:
        #     distance = source_distance

        # # calculate the similarity measure of each neighbor (m, seq)
        # comp_func = CompFunction(self.sigma[0], distance, self.type_compatibility_function[0], self.graph_label[0])
        # simi = comp_func.run()

        # #print('simi.shape, self.kernel.shape', simi.shape, self.kernel.shape)
        # # calculates the weights associated with each neighbor (m, seq)
        # # weight = K.dot(simi, self.kernel)
        # weight = simi
        # #weight = K.bias_add(weight, self.bias)
        # #weight = K.softmax(weight)

        # # repeats the previous vector as many times as the feature number plus the point target and features extras
        # # (input_phenomenon + 1 + num_features_extras, seq)
        # prob_repeat = RepeatVector(self.shape_input_phenomenon[0] + self.num_features_extras[0])(weight)

        # print('prob_repeat', prob_repeat.shape, 'node2vec', node2vec.shape)

        # # inverts the dimensions in such a way that in each line,
        # # we have the weight assigned to the neighbor (seq, input_phenomenon + 1 + num_features_extras)
        # prob_repeat = Permute((2, 1))(prob_repeat)

        # # multiplies each neighbor's feature by its respective weight
        # # (seq, input_phenomenon + 1 + num_features_extras) x (seq, input_phenomenon + 1 + num_features_extras)
        # relevance = multiply([prob_repeat, context])

        # # add each column to find the mean vector (input_phenomenon + 1 + num_features_extras,)
        # mean = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), name=self.suffix_mean[0])(relevance)

        # print('mean.shape', mean.shape)

        # return tf.concat((att_output, mean), -1) # mean

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
