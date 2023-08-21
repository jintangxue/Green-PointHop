import argparse
import modelnet_data
import pointhop
import sklearn
import aggregation_utils
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

    print(train_data.shape, train_label.shape, valid_data.shape, valid_label.shape)

    feat_valid = []
    feat_valid_final = []
    LLSR = True

    with open(os.path.join(LOG_DIR, 'params.pkl'), 'rb') as f:
        params_total = pickle.load(f)
    params = params_total['params:', 0]
    weight = params_total['weight:', 0]

    # Octant feature
    log_string('------------Test {} --------------'.format(0))
    leaf_node_test, hop_node_test_octant, test_new_data = pointhop.pointhop_pred(False, valid_data,
                                                                                 pca_params=params,
                                                                                 n_newpoint=num_point,
                                                                                 n_sample=[num_sample[0]])

    print("hop_node_test_octant:", len(num_point))
    log_string('Octant shape: {}'.format(np.array(hop_node_test_octant[0]).shape))
    time_find_hop = time.time()
    log_string('Find hop is {} minutes'.format((time_find_hop - time_start) // 60))

    # Aggregation
    # Global Aggregation
    feature_test_temp = aggregation_utils.global_agg(hop_node_test_octant)
    time_find_hist_global = time.time()
    log_string('Find global agg is {} minutes'.format((time_find_hist_global - time_find_hop) // 60))

    # Cone Aggregation
    feature_test_cones = aggregation_utils.six_cones_agg(test_new_data, hop_node_test_octant, alpha=alpha)
    time_find_hist_cone = time.time()
    log_string('Find cone agg is {} minutes'.format((time_find_hist_cone - time_find_hist_global) // 60))

    print("feature_test_temp:", len(feature_test_temp))
    print("feature_test_cones:", len(feature_test_cones))
    feature_test_one = []
    for j in range(len(num_point)):
        print("feature_test_temp[{}]:".format(j), np.array(feature_test_temp[j]).shape)
        print("feature_test_cones[{}]:".format(j), np.array(feature_test_cones[j]).shape)

        N_cones = np.array(feature_test_cones[j]).shape[0]

        feature_test_cones_1 = np.array(feature_test_cones[j]).reshape(N_cones,
                                                                         np.array(feature_test_temp[j]).shape[
                                                                             0],
                                                                         -1)

        print("feature_test_temp:", np.array(feature_test_temp[j]).shape)
        print("feature_test_cones:", np.array(feature_test_cones_1).shape)
        feature_test_temp_1 = np.concatenate([[feature_test_temp[j]], feature_test_cones_1], axis=0)
        feature_test_one.append(feature_test_temp_1)

    feature_test_temp = feature_test_one

    # DFT
    feature_test_octant = [np.concatenate(feature_test_temp[0], axis=-1)]

    print("feature_test_octant:", feature_test_octant[0].shape)
    feature_test = np.array(feature_test_octant[0])
    print("feature_test:", feature_test.shape)
    ind = params_total['DFT_ind:', 0]
    feature_test = feature_test[:, ind]
    
    time_DFT = time.time()
    log_string('DFT is {} minutes'.format((time_DFT - time_find_hist_cone) // 60))

    print("feature_test:", feature_test.shape)

    # LLSR Classifier
    feature_valid, pred_valid = pointhop.llsr_pred(feature_test, weight)
    feat_valid_final.append(feature_valid)
    acc_valid = sklearn.metrics.accuracy_score(valid_label, pred_valid)
    acc = pointhop.average_acc(valid_label, pred_valid)
    log_string('test: {} , test mean: {}'.format(acc_valid, np.mean(acc)))
    log_string('per-class: {}'.format(acc))

    time_end = time.time()
    log_string('totally time cost is {} minutes'.format((time_end - time_start) / 60))
    log_string('totally time cost is {} minutes'.format((time_end - time_start) // 60))

    with open(os.path.join(LOG_DIR, 'params.pkl'), 'wb') as f:
        pickle.dump(params_total, f)


if __name__ == '__main__':
    main()
