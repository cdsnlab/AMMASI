
from tensorflow.keras.layers import Input, RepeatVector


def getnode2vec(dim_node2vec, num_nearest_geo, num_nearest_eucli):

    """

    :param shape_node2vec:
    :param num_nearest_geo:
    :param num_nearest_eucli:
    :return:
    """

    ######################## Input of phenomeno ########################

    # the phenomena input features X_train.shape[1] (m, features)
    input_geo_n2v = Input(shape=(1+num_nearest_geo, dim_node2vec,), name='input_geo_n2v')
    input_eucli_n2v = Input(shape=(1+num_nearest_eucli, dim_node2vec,), name='input_eucli_n2v')

    ######################################################################

    return input_geo_n2v, input_eucli_n2v
