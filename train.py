import argparse
import modelnet_data
import pointhop
import sklearn
import aggregation_utils
import feat_utils
import os
import time
import pickle
import numpy as np

np.set_printoptions(threshold=np.inf)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

parser = argparse.ArgumentParser()
parser.add_argument('--initial_point', type=int, default=1024, help='Point Number [256/512/1024/2048]')
parser.add_argument('--validation', default=False, help='Split train data or not')
parser.add_argument('--num_features', default=1569, help='Feature Number after DFT')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', default=[1024], help='Point Number after down sampling')
parser.add_argument('--num_sample', default=[32], help='KNN query number')
parser.add_argument('--threshold', default=0.0001, help='threshold')
parser.add_argument('--alpha', default=np.pi * 150 / 180, help='angle of cones')
FLAGS = parser.parse_args()

initial_point = FLAGS.initial_point
VALID = FLAGS.validation
num_features = FLAGS.num_features
num_point = FLAGS.num_point
num_sample = FLAGS.num_sample
threshold = FLAGS.threshold
alpha = FLAGS.alpha

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def main():
    time_start = time.time()
    # load data
    train_data, train_label = modelnet_data.data_load(num_point=initial_point,
                                                      data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'),
                                                      train=True)
    test_data, test_label = modelnet_data.data_load(num_point=initial_point,
                                                    data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'),
                                                    train=False)

    # validation set
    if VALID:
        train_data, train_label, valid_data, valid_label = modelnet_data.data_separate(train_data, train_label)
    else:
        valid_data = test_data
        valid_label = test_label

    params_total = {}
    feat_train = []

    # Octant features
    log_string('-------------------Getting octant features-------------------')
    params, hop_node_train_octant, leaf_node, leaf_node_energy, train_new_data = pointhop.pointhop_train(True,
                                                                                                         train_data,
                                                                                                         n_newpoint=num_point,
                                                                                                         n_sample=[num_sample[0]],
                                                                                                         threshold=threshold)

    params_total['params:', 0] = params
    time_find_hop = time.time()
    log_string('Find octant features: {} minutes'.format((time_find_hop - time_start) // 60))

    # Aggregation
    log_string('-------------------Aggregation-------------------')
    # Global Aggregation
    feature_train_temp = aggregation_utils.global_agg(hop_node_train_octant)
    time_find_hist_global = time.time()
    log_string('Find global agg: {} minutes'.format((time_find_hist_global - time_find_hop) // 60))

    # Cone Aggregation
    feature_train_cones = aggregation_utils.six_cones_agg(train_new_data, hop_node_train_octant, alpha=alpha)
    time_find_hist_cone = time.time()
    log_string('Find cone agg: {} minutes'.format((time_find_hist_cone - time_find_hist_global) // 60))

    # print("feature_train_temp:", len(feature_train_temp))
    # print("feature_train_cones:", len(feature_train_cones))
    feature_train_one = []
    for j in range(len(num_point)):
        # print("feature_train_temp[{}]:".format(j), np.array(feature_train_temp[j]).shape)
        # print("feature_train_cones[{}]:".format(j), np.array(feature_train_cones[j]).shape)

        N_cones = np.array(feature_train_cones[j]).shape[0]

        feature_train_cones_temp = np.array(feature_train_cones[j]).reshape(N_cones,
                                                                         np.array(feature_train_temp[j]).shape[
                                                                             0], -1)

        # print("feature_train_temp:", np.array(feature_train_temp[j]).shape)
        # print("feature_train_cones:", np.array(feature_train_cones_temp).shape)
        feature_train_temp_1 = np.concatenate([[feature_train_temp[j]], feature_train_cones_temp], axis=0)

        feature_train_one.append(feature_train_temp_1)
        feature_train_temp = feature_train_one

    # DFT
    log_string('-------------------DFT-------------------')
    feature_train_octant = [np.concatenate(feature_train_temp[0], axis=-1)]

    # print("feature_train_octant:", feature_train_octant[0].shape)
    feature_train = np.array(feature_train_octant[0])
    # print("feature_train:", feature_train.shape)

    ind, _ = feat_utils.feature_selection(feature_train, train_label, 0)

    ind = ind[:int(len(ind) * num_features / len(ind))]
    params_total['DFT_ind:', 0] = ind
    # print("feature_train", feature_train.shape)
    feature_train = feature_train[:, ind]
    
    time_DFT = time.time()
    log_string('DFT is {} minutes'.format((time_DFT - time_find_hist_cone) // 60))

    # print("feature_train:", feature_train.shape)
    
    # LLSR Classifier
    log_string('-------------------LLSR Classifier-------------------')
    weight = pointhop.llsr_train(feature_train, train_label)
    params_total['weight:', 0] = weight
    feature_train, pred_train = pointhop.llsr_pred(feature_train, weight)
    feat_train.append(feature_train)

    acc_train = sklearn.metrics.accuracy_score(train_label, pred_train)
    log_string('train accuracy: {}'.format(acc_train))

    time_end = time.time()
    log_string('totally time cost is {} minutes'.format((time_end - time_start) / 60))
    log_string('totally time cost is {} minutes'.format((time_end - time_start) // 60))

    with open(os.path.join(LOG_DIR, 'params.pkl'), 'wb') as f:
        pickle.dump(params_total, f)


if __name__ == '__main__':
    main()
