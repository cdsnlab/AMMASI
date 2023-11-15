import os
import numpy as np

def detect_category(samples, max_category):
    uniques = list(np.unique(samples))
    if len(uniques) < max_category:
        return uniques
    else:
        return []
    

def load_data_ours(args):
    metadata = dict(args=args)
    print(metadata)
    data = np.load(f'../datasets/processed/{args.dataset}/processed_data_poi.npz')
    
    Train_feat, Train_latlon, Train_poidist, Train_price = data['Train_feat'], data['Train_latlon'], data['Train_poidist'], data['Train_price']
    Test_feat, Test_latlon, Test_poidist, Test_price = data['Test_feat'], data['Test_latlon'], data['Test_poidist'], data['Test_price']

    beta = {'fc': 0.045, 'kc':0.035, 'sp': 0.020, 'poa': 0.025}
    Train_poiprox = np.exp(-(Train_poidist / beta[args.dataset])**2 / 2)
    Test_poiprox = np.exp(-(Test_poidist / beta[args.dataset])**2 / 2)

    metadata['grid_emb'] = np.load(f'../datasets/{args.dataset}/grid_vectors_gaussian.npy').astype(np.float32)

    miny = min(np.min(Train_latlon[:, 0]), np.min(Test_latlon[:, 0])) - 0.01
    maxy = max(np.max(Train_latlon[:, 0]), np.max(Test_latlon[:, 0])) + 0.01
    minx = min(np.min(Train_latlon[:, 1]), np.min(Test_latlon[:, 1])) - 0.01
    maxx = max(np.max(Train_latlon[:, 1]), np.max(Test_latlon[:, 1])) + 0.01

    ncols = 100  #1000
    nrows = 100  #int(1200 * (maxy - miny) / (maxx - minx))
    width = (maxx - minx) / ncols
    height = (maxy - miny) / nrows

    metadata['miny'] = miny
    metadata['maxy'] = maxy
    metadata['minx'] = minx
    metadata['maxx'] = maxx

    metadata['ncols'] = ncols
    metadata['nrows'] = nrows
    metadata['width'] = width
    metadata['height'] = height

    Train_i = ((Train_latlon[:, 1] - minx) / width).astype(int)
    Train_j = ((Train_latlon[:, 0] - miny) / height).astype(int)
    Train_ij = np.stack((Train_i, Train_j), -1)

    Test_i = ((Test_latlon[:, 1] - minx) / width).astype(int)
    Test_j = ((Test_latlon[:, 0] - miny) / height).astype(int)
    Test_ij = np.stack((Test_i, Test_j), -1)

    # thres_num_neighbors = 0
    # if args.dataset == 'fc':
    #     thres_num_neighbors = 20
    # if args.dataset == 'kc':
    #     thres_num_neighbors = 45
    # if args.dataset == 'sp':
    #     thres_num_neighbors = 60
    # if args.dataset == 'poa':
    #     thres_num_neighbors = 30
    
    # Nidx_train, Ndist_train = data['Train_idx_geo'][:, :thres_num_neighbors], data['Train_dist_geo'][:, :thres_num_neighbors]
    # Nidx_test, Ndist_test = data['Test_idx_geo'][:, :thres_num_neighbors], data['Test_dist_geo'][:, :thres_num_neighbors]

    Nidx_train, Ndist_train = data['Train_idx_geo'], data['Train_dist_geo']
    Nidx_test, Ndist_test = data['Test_idx_geo'], data['Test_dist_geo']

    Eidx_train, Edist_train = data['Train_idx_eucli'], data['Train_dist_eucli']
    Eidx_test, Edist_test = data['Test_idx_eucli'], data['Test_dist_eucli']

    # print('Eidx_test, Edist_test', Eidx_test.shape, Edist_test.shape)

    metadata['num_neighbors'] = Nidx_train.shape[1]
    metadata['max_neighboridx'] = Train_feat.shape[0]


    # Feature Construction
    X_train = Train_feat
    X_test = Test_feat
    # print('args.use_areaemb', args.use_areaemb)
    # print('args.use_poiprox', args.use_poiprox)
        
    if args.use_poiprox:
        X_train = np.concatenate((X_train, Train_poiprox), -1)
        X_test = np.concatenate((X_test, Test_poiprox), -1)
    if args.use_locfeat:
        X_train = np.concatenate((Train_latlon, X_train), -1)
        X_test = np.concatenate((Test_latlon, X_test), -1)

    X_mean = np.mean(X_train, 0)
    X_std = np.std(X_train, 0) + 0.1
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std


    metadata['poi_features'] = Train_poiprox.shape[1]
    
    y_mean = np.mean(Train_price)
    y_std = np.std(Train_price)
    metadata['y_mean'] = y_mean
    metadata['y_std'] = y_std

    y_train = Train_price
    y_test = Test_price

    metadata['Train_features'] = np.concatenate((X_train, (Train_price - y_mean) / y_std), -1)

    ##
    X_train = np.concatenate((Train_ij, X_train), -1)
    X_test = np.concatenate((Test_ij, X_test), -1)
    metadata['num_features'] = X_train.shape[1]

    dataset = X_train, Nidx_train, Ndist_train, Eidx_train, Edist_train, y_train, \
              X_test , Nidx_test , Ndist_test , Eidx_test, Edist_test, y_test
    return dataset, metadata

 