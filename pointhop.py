import math
import sklearn
import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA
import point_utils
import threading
from sklearn.cluster import KMeans
import feat_utils


def sample_knn(point_data, n_newpoint, n_sample):
    point_num = point_data.shape[1]
    if n_newpoint == point_num:
        new_xyz = point_data
    else:
        new_xyz = point_utils.furthest_point_sample(point_data, n_newpoint)
    idx = point_utils.knn(new_xyz, point_data, n_sample)
    return new_xyz, idx


def tree(Train, Bias, point_data, data, grouped_feature, idx, pre_energy, threshold, params):
    if grouped_feature is None:
        grouped_feature = data
    grouped_feature = point_utils.gather_fea(idx, point_data, grouped_feature)  # (B, 1024, 24)
    s1 = grouped_feature.shape[0]
    s2 = grouped_feature.shape[1]
    grouped_feature = grouped_feature.reshape(s1 * s2, -1)  # (B*1024, 24)

    if Train is True:
        kernels, mean, energy = find_kernels_pca(grouped_feature)
        bias = LA.norm(grouped_feature, axis=1)
        bias = np.max(bias)
        if pre_energy is not None:
            energy = energy * pre_energy
        num_node = np.sum(energy > threshold)  # number of nodes go to next step
        params = {}
        params['bias'] = bias
        params['kernel'] = kernels
        params['pca_mean'] = mean
        params['energy'] = energy
        params['num_node'] = num_node
    else:
        kernels = params['kernel']
        mean = params['pca_mean']
        bias = params['bias']

    if Bias is True:
        grouped_feature = grouped_feature + bias

    transformed = np.matmul(grouped_feature, np.transpose(kernels))  # (B*1024, 24)

    if Bias is True:
        e = np.zeros((1, kernels.shape[0]))
        e[0, 0] = 1
        transformed -= bias * e

    transformed = transformed.reshape(s1, s2, -1)  # (B, 1024, 24)
    output = []
    for i in range(transformed.shape[-1]):
        output.append(transformed[:, :, i].reshape(s1, s2, 1))  # all layers after Saab (24, B, 1024, 1)
    return params, output


def tree_multi(Train, Bias, point_data, data, grouped_feature, idx, pre_energy, threshold, params, j, index, params_t, out):
    if grouped_feature is None:
        grouped_feature = data
    grouped_feature = point_utils.gather_fea(idx, point_data, grouped_feature)
    s1 = grouped_feature.shape[0]
    s2 = grouped_feature.shape[1]
    grouped_feature = grouped_feature.reshape(s1 * s2, -1)

    if Train is True:
        kernels, mean, energy = find_kernels_pca(grouped_feature)
        bias = LA.norm(grouped_feature, axis=1)
        bias = np.max(bias)
        if pre_energy is not None:
            energy = energy * pre_energy
        num_node = np.sum(energy > threshold)
        params = {}
        params['bias'] = bias
        params['kernel'] = kernels
        params['pca_mean'] = mean
        params['energy'] = energy
        params['num_node'] = num_node
    else:
        kernels = params['kernel']
        mean = params['pca_mean']
        bias = params['bias']

    if Bias is True:
        grouped_feature = grouped_feature + bias

    transformed = np.matmul(grouped_feature, np.transpose(kernels))

    if Bias is True:
        e = np.zeros((1, kernels.shape[0]))
        e[0, 0] = 1
        transformed -= bias * e

    transformed = transformed.reshape(s1, s2, -1)
    output = []
    for i in range(transformed.shape[-1]):
        output.append(transformed[:, :, i].reshape(s1, s2, 1))
    index.append(j)
    params_t.append(params)
    out.append(output)


def pointhop_train(Train, data, n_newpoint, n_sample, threshold):
    '''
    Train based on the provided samples.
    :param train_data: [num_samples, num_point, feature_dimension]
    :param n_newpoint: point numbers used in every stage
    :param n_sample: k nearest neighbors
    :param layer_num: num kernels to be preserved
    :param energy_percent: the percent of energy to be preserved
    :return: idx, new_idx, final stage feature, feature, pca_params
    '''

    point_data = data
    Bias = [False, True, True, True]
    info = {}
    pca_params = {}
    hop_node = []
    leaf_node = []
    leaf_node_energy = []
    train_new_data = []

    for i in range(len(n_newpoint)):
        new_xyz, idx = sample_knn(point_data, n_newpoint[i], n_sample[i])
        if i == 0:
            print(i)
            pre_energy = 1
            params, output = tree(Train, Bias[i], point_data, data, None, idx, pre_energy, threshold, None)
            pca_params['Layer_{:d}_pca_params'.format(i)] = params
            num_node = params['num_node']
            energy = params['energy']
            info['Layer_{:d}_feature'.format(i)] = output[:num_node]
            info['Layer_{:d}_energy'.format(i)] = energy
            info['Layer_{:d}_num_node'.format(i)] = num_node
            if num_node != len(output):
                for m in range(num_node, len(output), 1):
                    leaf_node.append(output[m])
                    leaf_node_energy.append(energy[m])
            # eigen_feature = point_utils.gather_eigen_fea(idx, point_data)  # (B, 1024, 24)
            # eigen_feature = eigen_feature.reshape(-1, len(data), n_newpoint[i], 1)
            # hop_node.append(np.concatenate([output, eigen_feature], axis=0))  # output: (24, B, 1024, 1)
            hop_node.append(output)
        elif i == 1:
            output = info['Layer_{:d}_feature'.format(i - 1)]
            pre_energy = info['Layer_{:d}_energy'.format(i - 1)]
            num_node = info['Layer_{:d}_num_node'.format(i - 1)]
            s1 = 0
            index = []
            params_t = []
            out = []
            threads = []
            hop_node_t = []
            for j in range(num_node):
                threads.append(threading.Thread(target=tree_multi, args=(Train, Bias[i], point_data, data, output[j], idx,
                                                                   pre_energy[j], threshold, None, j, index, params_t, out)))
            for t in threads:
                t.setDaemon(False)
                t.start()
            for t in threads:
                if t.isAlive():
                    t.join()
            # print("num_node:", num_node)
            for j in range(num_node):  # num nodes from last hop
                print(i, j)
                ind = np.where(np.array(index) == j)[0]
                params = params_t[ind[0]]
                output_t = out[ind[0]]
                pca_params['Layer_{:d}_{:d}_pca_params'.format(i, j)] = params
                num_node_t = params['num_node']
                energy = params['energy']
                info['Layer_{:d}_{:d}_feature'.format(i, j)] = output_t[:num_node_t]
                info['Layer_{:d}_{:d}_energy'.format(i, j)] = energy
                info['Layer_{:d}_{:d}_num_node'.format(i, j)] = num_node_t
                s1 = s1 + num_node_t
                if num_node_t != len(output_t):  # output_t: (8, 50, 128, 1)
                    for m in range(num_node_t, len(output_t), 1):
                        leaf_node.append(output_t[m])
                        leaf_node_energy.append(energy[m])
                hop_node_t.append(output_t)
            hop_node_t = np.concatenate(hop_node_t, axis=0)  # (num_node*8, 50, 128, 1)
            hop_node.append(hop_node_t)

        elif i == 2:
            num_node = info['Layer_{:d}_num_node'.format(i - 2)]  # num node in the first layer
            hop_node_t = []
            for j in range(num_node):
                output = info['Layer_{:d}_{:d}_feature'.format(i - 1, j)]
                pre_energy = info['Layer_{:d}_{:d}_energy'.format(i - 1, j)]
                num_node_t = info['Layer_{:d}_{:d}_num_node'.format(i - 1, j)]

                index = []
                params_t = []
                out = []
                threads = []
                for k in range(num_node_t):
                    threads.append(
                        threading.Thread(target=tree_multi, args=(Train, Bias[i], point_data, data, output[k], idx,
                                                              pre_energy[k], threshold, None, k, index, params_t, out)))
                for t in threads:
                    t.setDaemon(False)
                    t.start()
                for t in threads:
                    if t.isAlive():
                        t.join()
                # print(num_node_t)
                for k in range(num_node_t):
                    print(i, j, k)
                    ind = np.where(np.array(index) == k)[0]
                    params = params_t[ind[0]]
                    output_t = out[ind[0]]
                    pca_params['Layer_{:d}_{:d}_{:d}_pca_params'.format(i, j, k)] = params
                    num_node_tt = params['num_node']
                    energy = params['energy']
                    info['Layer_{:d}_{:d}_{:d}_feature'.format(i, j, k)] = output_t[:num_node_tt]
                    info['Layer_{:d}_{:d}_{:d}_energy'.format(i, j, k)] = energy
                    info['Layer_{:d}_{:d}_{:d}_num_node'.format(i, j, k)] = num_node_tt
                    if num_node_tt != len(output_t):
                        for m in range(num_node_tt, len(output_t), 1):
                            leaf_node.append(output_t[m])
                            leaf_node_energy.append(energy[m])
                    hop_node_t.append(output_t)
            hop_node_t = np.concatenate(hop_node_t, axis=0)  # (num_node_sum*8, 50, 128, 1)
            # print(hop_node_t.shape)
            hop_node.append(hop_node_t)

        elif i == 3:
            num_node = info['Layer_{:d}_num_node'.format(i - 3)]
            hop_node_t = []
            for j in range(num_node):
                num_node_t = info['Layer_{:d}_{:d}_num_node'.format(i - 2, j)]
                for k in range(num_node_t):
                    output = info['Layer_{:d}_{:d}_{:d}_feature'.format(i - 1, j, k)]
                    pre_energy = info['Layer_{:d}_{:d}_{:d}_energy'.format(i - 1, j, k)]
                    num_node_tt = info['Layer_{:d}_{:d}_{:d}_num_node'.format(i - 1, j, k)]

                    index = []
                    params_t = []
                    out = []
                    threads = []
                    for t in range(num_node_tt):
                        threads.append(
                            threading.Thread(target=tree_multi, args=(Train, Bias[i], point_data, data, output[t], idx,
                                                                  pre_energy[t], threshold, None, t, index, params_t, out)))
                    for t in threads:
                        t.setDaemon(False)
                        t.start()
                    for t in threads:
                        if t.isAlive():
                            t.join()

                    for t in range(num_node_tt):
                        print(i, j, k, t)
                        ind = np.where(np.array(index) == t)[0]
                        params = params_t[ind[0]]
                        output_t = out[ind[0]]
                        pca_params['Layer_{:d}_{:d}_{:d}_{:d}_pca_params'.format(i, j, k, t)] = params
                        num_node_ttt = params['num_node']
                        energy = params['energy']
                        info['Layer_{:d}_{:d}_{:d}_{:d}_feature'.format(i, j, k, t)] = output_t[:num_node_ttt]
                        info['Layer_{:d}_{:d}_{:d}_{:d}_energy'.format(i, j, k, t)] = energy
                        info['Layer_{:d}_{:d}_{:d}_{:d}_num_node'.format(i, j, k, t)] = num_node_ttt
                        for m in range(len(output_t)):
                            leaf_node.append(output_t[m])
                            leaf_node_energy.append(energy[m])
                        hop_node_t.append(output_t)
            hop_node_t = np.concatenate(hop_node_t, axis=0)  # (num_node_sum*8, 50, 64, 1)
            # print(hop_node_t.shape)
            hop_node.append(hop_node_t)
        point_data = new_xyz
        train_new_data.append(new_xyz)
    # print(len(leaf_node))
    return pca_params, hop_node, leaf_node, leaf_node_energy, train_new_data


def pointhop_pred(Train, data, pca_params, n_newpoint, n_sample):
    '''
    Test based on the provided samples.
    :param test_data: [num_samples, num_point, feature_dimension]
    :param pca_params: pca kernel and mean
    :param n_newpoint: point numbers used in every stage
    :param n_sample: k nearest neighbors
    :param layer_num: num kernels to be preserved
    :param idx_save: knn index
    :param new_xyz_save: down sample index
    :return: final stage feature, feature, pca_params
    '''

    point_data = data
    Bias = [False, True, True, True]
    info_test = {}
    leaf_node = []
    hop_node = []
    test_new_data = []

    for i in range(len(n_newpoint)):
        new_xyz, idx = sample_knn(point_data, n_newpoint[i], n_sample[i])
        if i == 0:
            print(i)
            params = pca_params['Layer_{:d}_pca_params'.format(i)]
            num_node = params['num_node']
            params_t, output = tree(Train, Bias[i], point_data, data, None, idx, None, None, params)
            info_test['Layer_{:d}_feature'.format(i)] = output[:num_node]
            info_test['Layer_{:d}_num_node'.format(i)] = num_node
            if num_node != len(output):
                for m in range(num_node, len(output), 1):
                    leaf_node.append(output[m])

            hop_node.append(output)  # output: (24, B, 1024, 1)

        elif i == 1:
            output = info_test['Layer_{:d}_feature'.format(i - 1)]
            num_node = info_test['Layer_{:d}_num_node'.format(i - 1)]

            index = []
            params_t = []
            out = []
            threads = []
            hop_node_t = []
            for j in range(num_node):
                threads.append(
                    threading.Thread(target=tree_multi, args=(Train, Bias[i], point_data, data, output[j], idx,
                                                              None, None, pca_params['Layer_{:d}_{:d}_pca_params'.format(i, j)], j, index, params_t, out)))
            for t in threads:
                t.setDaemon(False)
                t.start()
            for t in threads:
                if t.isAlive():
                    t.join()

            for j in range(num_node):
                print(i, j)
                ind = np.where(np.array(index) == j)[0]
                output_t = out[ind[0]]
                params = pca_params['Layer_{:d}_{:d}_pca_params'.format(i, j)]
                num_node_t = params['num_node']
                info_test['Layer_{:d}_{:d}_feature'.format(i, j)] = output_t[:num_node_t]
                info_test['Layer_{:d}_{:d}_num_node'.format(i, j)] = num_node_t
                if num_node_t != len(output_t):
                    for m in range(num_node_t, len(output_t), 1):
                        leaf_node.append(output_t[m])
                hop_node_t.append(output_t)
            hop_node_t = np.concatenate(hop_node_t, axis=0)  # (num_node*8, 50, 128, 1)
            hop_node.append(hop_node_t)
        elif i == 2:
            num_node = info_test['Layer_{:d}_num_node'.format(i - 2)]
            hop_node_t = []
            for j in range(num_node):
                output = info_test['Layer_{:d}_{:d}_feature'.format(i - 1, j)]
                num_node_t = info_test['Layer_{:d}_{:d}_num_node'.format(i - 1, j)]

                index = []
                params_t = []
                out = []
                threads = []
                for k in range(num_node_t):
                    threads.append(
                        threading.Thread(target=tree_multi, args=(Train, Bias[i], point_data, data, output[k], idx,
                                                              None, None, pca_params['Layer_{:d}_{:d}_{:d}_pca_params'.format(i, j, k)], k, index, params_t, out)))
                for t in threads:
                    t.setDaemon(False)
                    t.start()
                for t in threads:
                    if t.isAlive():
                        t.join()
                for k in range(num_node_t):
                    print(i, j, k)
                    params = pca_params['Layer_{:d}_{:d}_{:d}_pca_params'.format(i, j, k)]
                    num_node_tt = params['num_node']
                    ind = np.where(np.array(index) == k)[0]
                    output_t = out[ind[0]]
                    info_test['Layer_{:d}_{:d}_{:d}_feature'.format(i, j, k)] = output_t[:num_node_tt]
                    info_test['Layer_{:d}_{:d}_{:d}_num_node'.format(i, j, k)] = num_node_tt
                    if num_node_tt != len(output_t):
                        for m in range(num_node_tt, len(output_t), 1):
                            leaf_node.append(output_t[m])
                    hop_node_t.append(output_t)
            hop_node_t = np.concatenate(hop_node_t, axis=0)  # (num_node*8, 50, 128, 1)
            hop_node.append(hop_node_t)

        elif i == 3:
            num_node = info_test['Layer_{:d}_num_node'.format(i - 3)]
            hop_node_t = []
            for j in range(num_node):
                num_node_t = info_test['Layer_{:d}_{:d}_num_node'.format(i - 2, j)]
                for k in range(num_node_t):
                    output = info_test['Layer_{:d}_{:d}_{:d}_feature'.format(i - 1, j, k)]
                    num_node_tt = info_test['Layer_{:d}_{:d}_{:d}_num_node'.format(i - 1, j, k)]

                    index = []
                    params_t = []
                    out = []
                    threads = []
                    for t in range(num_node_tt):
                        threads.append(
                            threading.Thread(target=tree_multi, args=(Train, Bias[i], point_data, data, output[t], idx,
                                                                  None, None, pca_params['Layer_{:d}_{:d}_{:d}_{:d}_pca_params'.format(i, j, k, t)], t, index, params_t, out)))
                    for t in threads:
                        t.setDaemon(False)
                        t.start()
                    for t in threads:
                        if t.isAlive():
                            t.join()
                    for t in range(num_node_tt):
                        print(i, j, k, t)
                        params = pca_params['Layer_{:d}_{:d}_{:d}_{:d}_pca_params'.format(i, j, k, t)]
                        num_node_ttt = params['num_node']
                        ind = np.where(np.array(index) == t)[0]
                        output_t = out[ind[0]]
                        info_test['Layer_{:d}_{:d}_{:d}_{:d}_feature'.format(i, j, k, t)] = output_t[:num_node_ttt]
                        info_test['Layer_{:d}_{:d}_{:d}_{:d}_num_node'.format(i, j, k, t)] = num_node_ttt
                        for m in range(len(output_t)):
                            leaf_node.append(output_t[m])
                        hop_node_t.append(output_t)
            hop_node_t = np.concatenate(hop_node_t, axis=0)  # (num_node*8, 50, 128, 1)
            hop_node.append(hop_node_t)
        point_data = new_xyz
        test_new_data.append(new_xyz)
    # print(len(leaf_node))
    return leaf_node, hop_node, test_new_data


def remove_mean(features, axis):
    '''
    Remove the dataset mean.
    :param features [num_samples,...]
    :param axis the axis to compute mean
    
    '''
    feature_mean = np.mean(features, axis=axis, keepdims=True)
    feature_remove_mean = features-feature_mean
    return feature_remove_mean, feature_mean


def remove_zero_patch(samples):
    std_var = (np.std(samples, axis=1)).reshape(-1, 1)
    ind_bool = (std_var == 0)
    ind = np.where(ind_bool==True)[0]
    samples_new = np.delete(samples, ind, 0)
    return samples_new


def find_kernels_pca(sample_patches):
    '''
    Do the PCA based on the provided samples.
    If num_kernels is not set, will use energy_percent.
    If neither is set, will preserve all kernels.
    :param samples: [num_samples, feature_dimension]
    :param num_kernels: num kernels to be preserved
    :param energy_percent: the percent of energy to be preserved
    :return: kernels, sample_mean
    '''
    # Remove patch mean
    sample_patches_centered, dc = remove_mean(sample_patches, axis=1)
    sample_patches_centered = remove_zero_patch(sample_patches_centered)
    # Remove feature mean (Set E(X)=0 for each dimension)
    training_data, feature_expectation = remove_mean(sample_patches_centered, axis=0)

    pca = PCA(n_components=training_data.shape[1], svd_solver='full', whiten=True)
    pca.fit(training_data)

    num_channels = sample_patches.shape[-1]
    largest_ev = [np.var(dc*np.sqrt(num_channels))]
    dc_kernel = 1/np.sqrt(num_channels)*np.ones((1, num_channels))/np.sqrt(largest_ev)

    kernels = pca.components_[:, :]
    mean = pca.mean_
    kernels = np.concatenate((dc_kernel, kernels), axis=0)[:kernels.shape[0], :]

    energy = np.concatenate((largest_ev, pca.explained_variance_[:kernels.shape[0]-1]), axis=0) \
             / (np.sum(pca.explained_variance_[:kernels.shape[0]-1]) + largest_ev)
    return kernels, mean, energy


def extract(feat):
    '''
    Do feature extraction based on the provided feature.
    :param feat: [num_layer, num_samples, feature_dimension]
    # :param pooling: pooling method to be used
    :return: feature
    '''
    mean = []
    maxi = []
    l1 = []
    l2 = []

    for i in range(len(feat)):
        mean.append(feat[i].mean(axis=1, keepdims=False))
        maxi.append(feat[i].max(axis=1, keepdims=False))
        l1.append(np.linalg.norm(feat[i], ord=1, axis=1, keepdims=False))
        l2.append(np.linalg.norm(feat[i], ord=2, axis=1, keepdims=False))
    mean = np.concatenate(mean, axis=-1)
    maxi = np.concatenate(maxi, axis=-1)
    l1 = np.concatenate(l1, axis=-1)
    l2 = np.concatenate(l2, axis=-1)

    return [mean, maxi, l1, l2]


def aggregate(feat, pool):
    feature = []
    for j in range(len(feat)):
        feature.append(feat[j] * pool[j])
    feature = np.concatenate(feature, axis=-1)
    return feature


def average_acc(label, pred_label):

    classes = np.arange(40)
    acc = np.zeros(len(classes))
    for i in range(len(classes)):
        ind = np.where(label == classes[i])[0]
        pred_test_special = pred_label[ind]
        acc[i] = len(np.where(pred_test_special == classes[i])[0])/float(len(ind))
    return acc


def onehot_encoding(n_class, labels):

    targets = labels.reshape(-1)
    one_hot_targets = np.eye(n_class)[targets]
    return one_hot_targets


def KMeans_Cross_Entropy(X, Y, num_class, num_bin=32):
    if np.unique(Y).shape[0] == 1:
        return 0
    if X.shape[0] < num_bin:
        return -1
    kmeans = KMeans(n_clusters=num_bin, random_state=0).fit(X)
    prob = np.zeros((num_bin, num_class))
    for i in range(num_bin):
        idx = (kmeans.labels_ == i)
        tmp = Y[idx]
        for j in range(num_class):
            prob[i, j] = float(tmp[tmp == j].shape[0]) / (float(Y[Y == j].shape[0]) + 1e-5)
    prob = (prob) / (np.sum(prob, axis=1).reshape(-1, 1) + 1e-5)
    true_indicator = onehot_encoding(num_class, Y)
    probab = prob[kmeans.labels_]
    return sklearn.metrics.log_loss(true_indicator, probab)/math.log(num_class)


def CE(X, Y, num_class):
    H = []
    for i in range(X.shape[1]):
        H.append(KMeans_Cross_Entropy(X[:, i].reshape(-1, 1), Y, num_class, num_bin=40))
    return np.array(H)


def llsr_train(feature, label):
    A = np.ones((feature.shape[0], 1))
    feature = np.concatenate((A, feature), axis=1)
    y = onehot_encoding(40, label)
    weight = np.matmul(LA.pinv(feature), y)
    return weight


def llsr_pred(feature, weight):
    A = np.ones((feature.shape[0], 1))
    feature = np.concatenate((A, feature), axis=1)
    feature = np.matmul(feature, weight)
    pred = np.argmax(feature, axis=1)
    return feature, pred


def new_extract(feat, num):
    '''
    num: how many points to choose
    Do feature extraction based on the provided feature.
    :param feat: [num_layer, num_batch, num_samples, feature_dimension(1)]
    :return: feature
    '''

    extracted_feat = []

    temp = feat[0]
    temp = np.squeeze(temp)
    temp = np.sort(temp, axis=1)
    K = temp.shape[1] - 1
    idx = (np.linspace(0, K, num) + 0.5).astype(int)
    for i in idx:
        extracted_feat.append([np.expand_dims(temp[:, i], axis=-1)])

    for i in range(1, len(feat)):  # (num_layer, B, K, 1)
        temp = feat[i]
        temp = np.squeeze(temp)
        temp = np.sort(temp, axis=1)
        K = temp.shape[1] - 1
        idx = (np.linspace(0, K, num) + 0.5).astype(int)
        for j in range(num):
            extracted_feat[j].append(np.expand_dims(temp[:, idx[j]], axis=-1))

    for i in range(len(extracted_feat)):
        extracted_feat[i] = np.concatenate(extracted_feat[i], axis=-1)

    return extracted_feat


def separate_agg(hop_node):
    [hop_node_1, hop_node_2, hop_node_3, hop_node_4] = hop_node
    feature_agg_1 = extract(hop_node_1)
    feature_agg_2 = extract(hop_node_2)
    feature_agg_3 = extract(hop_node_3)
    feature_agg_4 = extract(hop_node_4)

    feature_agg_1 = np.concatenate(feature_agg_1, axis=-1)
    feature_agg_2 = np.concatenate(feature_agg_2, axis=-1)
    feature_agg_3 = np.concatenate(feature_agg_3, axis=-1)
    feature_agg_4 = np.concatenate(feature_agg_4, axis=-1)
    print(feature_agg_1.shape)

    return [feature_agg_1, feature_agg_2, feature_agg_3, feature_agg_4]


def DFT_fs(Learn_DFT, feature, train_label, DFT_threshold, num_fea=2):
    fe_ind_all = []
    for i in range(len(feature)):
        feature_agg = feature[i]
        if Learn_DFT:
            ind, _ = feat_utils.feature_selection(feature_agg, train_label, i)
            if num_fea == 2:
                np.save('ind_four{}.npy'.format(i), np.array(ind))
            else:
                np.save('ind{}_four{}.npy'.format(num_fea, i), np.array(ind))
        else:
            if num_fea == 2:
                ind = np.load('ind_four{}.npy'.format(i))
            else:
                ind = np.load('ind{}_four{}.npy'.format(num_fea, i))
        fe_ind = ind[:int(len(ind) * DFT_threshold[i])]
        fe_ind_all.append(fe_ind)

    return fe_ind_all

