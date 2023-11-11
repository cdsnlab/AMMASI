from asi_sm.attention_layer import Attention, AttentionGeo, AttentionNew
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate


class Interpolation:
    """

    """

    def __init__(self, geointerpolation, num_neuron, num_layers, size_embedded, shape_input_phenomenon,
                 input_phenomenon, context_struc_eucli_target, context_geo_target_dist, shape_input_phenomenon_eucl,
                 type_compat_funct_eucli, type_compat_funct_geo, num_features_extras_struct,
                 num_features_extras_geo, cal_dist_struct, cal_dist_geo, graph_label, dist_eucli, 
                 dist_geo, n2v_geo, n2v_eucli, input_phe_w_lat_long, geo, euclidean, activation, num_nearest_geo, num_nearest_eucli,
                 sigma: list = None, num_nearest: int = None):

        """

        :param geointerpolation:
        :param num_neuron:
        :param num_layers:
        :param size_embedded:
        :param shape_input_phenomenon:
        :param input_phenomenon:
        :param context_struc_eucli_target:
        :param context_geo_target_dist:
        :param shape_input_phenomenon_eucl:
        :param type_compat_funct_eucli:
        :param type_compat_funct_geo:
        :param num_features_extras_struct:
        :param num_features_extras_geo:
        :param cal_dist_struct:
        :param cal_dist_geo:
        :param graph_label:
        :param dist_eucli:
        :param activation:
        :param dist_geo:
        :param input_phe_w_lat_long:
        :param geo:
        :param euclidean:
        :param num_nearest_geo:
        :param num_nearest_eucli:
        :param sigma:
        :param num_nearest:
        :param n2v_geo:
        :param n2v_eucli:
        """

        self.interpolation = geointerpolation
        self.num_neuron = num_neuron
        self.num_layers = num_layers
        self.size_embedded = size_embedded
        self.sigma = sigma
        self.num_nearest = num_nearest
        self.shape_input_phenomenon = shape_input_phenomenon
        self.shape_input_phenomenon_eucl = shape_input_phenomenon_eucl
        self.input_phenomenon = input_phenomenon
        self.context_struc_eucli_target = context_struc_eucli_target
        self.context_geo_target_dist = context_geo_target_dist
        self.type_compat_funct_eucli = type_compat_funct_eucli
        self.type_compat_funct_geo = type_compat_funct_geo
        self.num_features_extras_struct = num_features_extras_struct
        self.num_features_extras_geo = num_features_extras_geo
        self.cal_dist_struct = cal_dist_struct
        self.cal_dist_geo = cal_dist_geo
        self.graph_label = graph_label
        self.dist_eucli = dist_eucli
        self.dist_geo = dist_geo
        self.n2v_geo = n2v_geo
        self.n2v_eucli = n2v_eucli
        self.input_phe_w_lat_long = input_phe_w_lat_long
        self.geo = geo
        self.euclidean = euclidean
        self.num_nearest_geo = num_nearest_geo
        self.num_nearest_eucli = num_nearest_eucli
        self.activation = activation

        self.choices = {
            'simple asi': self.simpleasi
        }

        # adjustment factor of the Gaussian curve
        if sigma:
            self.sigma_struct_eucli = sigma[0]
            self.sigma_geo = sigma[1]
        else:
            self.sigma_struct_eucli = None
            self.sigma_geo = None

    def run(self):

        choice = self.interpolation
        action = self.choices.get(choice)
        if action:

            interpolation = action()

        else:
            interpolation = "{0} is not a valid choice".format(choice)
            print(interpolation)

        return interpolation

    def simpleasi(self):

        print('self.dist_eucli.shape, self.context_struc_eucli_target.shape, self.n2v_eucli.shape', \
                self.dist_eucli.shape, self.context_struc_eucli_target.shape, self.n2v_eucli.shape)
        print('self.dist_geo.shape, self.context_geo_target_dist.shape, self.n2v_geo.shape', \
                self.dist_geo.shape, self.context_geo_target_dist.shape, self.n2v_geo.shape)

        ######################## Structural euclidean attention data ########################


        if self.euclidean:
            mean_struct_eucli = Attention(sigma=self.sigma_struct_eucli,  # compatibility function
                                    num_nearest=self.num_nearest_eucli,
                                    shape_input_phenomenon=self.shape_input_phenomenon,
                                    type_compatibility_function=self.type_compat_funct_eucli,  # compatibility function
                                    num_features_extras=self.num_features_extras_struct,
                                    calculate_distance=self.cal_dist_struct,  # distance
                                    graph_label=self.graph_label + '_weight_struct_eucli',  # label
                                    suffix_mean='mean_strucut_eucli'  # label
                                    )([self.dist_eucli, self.context_struc_eucli_target])

        ######################## Geo attention data ###############################

        if self.geo:
            # mean_geo = Attention(sigma=self.sigma_geo,  # compatibility function
            #                num_nearest=self.num_nearest_geo,
            #                shape_input_phenomenon=self.shape_input_phenomenon,
            #                type_compatibility_function=self.type_compat_funct_geo,  # compatibility function
            #                num_features_extras=self.num_features_extras_geo,
            #                calculate_distance=self.cal_dist_geo,  # distance
            #                graph_label=self.graph_label + '_weight_geo',  # label
            #                suffix_mean='mean_geo'  # label
            #                )([self.dist_geo, self.context_geo_target_dist])
            mean_geo = AttentionGeo(sigma=self.sigma_geo,  # compatibility function
                           num_nearest=self.num_nearest_geo,
                           shape_input_phenomenon=self.shape_input_phenomenon,
                           type_compatibility_function=self.type_compat_funct_geo,  # compatibility function
                           num_features_extras=self.num_features_extras_geo,
                           calculate_distance=self.cal_dist_geo,  # distance
                           graph_label=self.graph_label + '_weight_geo',  # label
                           suffix_mean='mean_geo'  # label
                           )([self.dist_geo, self.context_geo_target_dist, self.n2v_geo])
            mean_geo_new = AttentionNew(#num_neuron=self.num_neuron, 
                            sigma=self.sigma_geo,  # compatibility function
                           num_nearest=self.num_nearest_geo,
                           shape_input_phenomenon=self.shape_input_phenomenon,
                           type_compatibility_function=self.type_compat_funct_geo,  # compatibility function
                           num_features_extras=self.num_features_extras_geo,
                           calculate_distance=self.cal_dist_geo,  # distance
                           graph_label=self.graph_label + '_weight_geo',  # label
                           suffix_mean='mean_geo'  # label
                           )([self.input_phenomenon, self.dist_geo, self.context_geo_target_dist, self.n2v_geo])

        ######################## Input hiden layer ########################

        if self.geo and self.euclidean:
            # concatenated to the three entries (X_train.shape[2]) ( X_train.shape[2]+1)  (X_train.shape[2]+2)
            # concate = concatenate([self.input_phenomenon, mean_struct_eucli, mean_geo])
            # concate = concatenate([self.input_phenomenon, mean_struct_eucli, mean_geo, mean_geo_new])
            concate = concatenate([self.input_phenomenon, mean_geo, mean_geo_new])
            # concate = concatenate([self.input_phenomenon, mean_geo])
            # print('self.input_phenomenon.shape, mean_struct_eucli.shape, mean_geo.shape, concate.shape', \
            #         self.input_phenomenon.shape, mean_struct_eucli.shape, mean_geo.shape, concate.shape)
        elif self.geo:
            # concatenated to the three entries (X_train.shape[2]+2)
            concate = concatenate([self.input_phenomenon, mean_geo])
        elif self.euclidean:
            # concatenated to the three entries (X_train.shape[2]+1)
            concate = concatenate([self.input_phenomenon, mean_struct_eucli])
        else:
            # concatenated to the three entries (X_train.shape[2]+1)
            concate = self.input_phenomenon

        ######################## hiden layer ########################

        feedforward = Dense(self.num_neuron, activation='elu')(concate)
        for l in range(self.num_layers):
            feedforward = Dense(self.num_neuron, activation='elu', name='full_connection_' + str(1 + l))(feedforward)

        print('feedforward', feedforward.shape)

        ######################## Embbeding ########################

        embedded = Dense(self.size_embedded, activation='elu', name='embedded')(feedforward)

        return embedded
