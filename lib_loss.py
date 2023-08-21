import numpy as np


def cal_entropy_from_y(y_array, y_distribution, num_cls):
    """
    :param y_array: all labels
    :param y_distribution: class histogram
    :param num_cls: number of classes
    :return: H (i) (L,t)
    """

    prob = np.zeros(num_cls)
    for c in range(num_cls):
        prob[c] = np.sum(y_array==c)/y_distribution[c]
    prob = prob/np.sum(prob)

    prob_tmp = np.copy(prob)
    prob_tmp[prob_tmp==0] = 1
    tmp = np.sum(-1*prob*np.log(prob_tmp),axis=-1)

    return tmp/np.log(num_cls)


def cal_entropy(prob, num_cls=10):
    prob_tmp = np.copy(prob)
    prob_tmp[prob_tmp==0] = 1
    tmp = np.sum(-1*prob*np.log(prob_tmp),axis=-1)
    return tmp/np.log(num_cls)


def class_distribution(y, num_cls=10):
    distribution = np.zeros(num_cls)
    for c in range(num_cls):
        distribution[c] = y.tolist().count(c)
    return distribution


def cal_weighted_H(X, y, bound, num_cls=10): # weighted entropy
    '''
    :param X: samples with only one feature
    :param y: label
    :param bound: optimize threshold
    :param num_cls: number of classes
    :return: H (i) (t)
    '''
    classwise = class_distribution(y, num_cls=num_cls)

    # only two bins
    if np.sum(X<bound)==0 or np.sum(X>=bound)==0:
        wH = 1
    else:
        left_y = y[X<bound]
        right_y = y[X>=bound]
        left_num = left_y.size
        right_num = right_y.size

        entropy = np.array([cal_entropy_from_y(left_y, classwise, num_cls), cal_entropy_from_y(right_y, classwise, num_cls)]).reshape(-1,1)

        num = np.array([left_num, right_num]).reshape(1,-1)
        num = num/np.sum(num,keepdims=1)

        wH = num @ entropy
    return wH

