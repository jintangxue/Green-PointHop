import numpy as np
import threading
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects


def get_global_agg(octant_feature, octant_data_shape, agg_results):
    # 7 aggregations
    agg_temp = []
    for i in range(octant_feature.shape[-1]):
        octant_one_feature = octant_feature[:, i].reshape(-1, 1)

        octant_one_feature = octant_one_feature.reshape(octant_data_shape[0], octant_data_shape[1], 1)  # (B, N, 1)

        agg_one_temp = []
        agg_one_temp.append(np.mean(octant_one_feature, axis=1))
        agg_one_temp.append(np.max(octant_one_feature, axis=1))
        agg_one_temp.append(np.min(octant_one_feature, axis=1))
        agg_one_temp.append(np.var(octant_one_feature, axis=1))
        agg_one_temp.append(np.std(octant_one_feature, axis=1))
        agg_one_temp.append(np.linalg.norm(octant_one_feature, ord=1, axis=1, keepdims=False))
        agg_one_temp.append(np.linalg.norm(octant_one_feature, ord=2, axis=1, keepdims=False))
        train_hist_one = np.concatenate(agg_one_temp, axis=-1)

        train_hist_one = np.array(train_hist_one)
        agg_temp.append(train_hist_one)

    # print("train_hist_temp", np.array(agg_temp).shape)

    agg_results.append(agg_temp)


def global_agg_multi(octant_fea):
    print("getting global aggregation")
    octant_fea_shape = octant_fea.shape

    octant_fea = octant_fea.reshape(-1, octant_fea.shape[-1])  # (512000, 33)

    # print("feature:", octant_fea.shape)
    batch_size = octant_fea.shape[-1] // 8
    agg_1 = []
    agg_2 = []
    agg_3 = []
    agg_4 = []
    agg_5 = []
    agg_6 = []
    agg_7 = []
    agg_8 = []
    threads = []
    t1 = threading.Thread(target=get_global_agg, args=(octant_fea[:, :batch_size], octant_fea_shape, agg_1))
    threads.append(t1)
    t2 = threading.Thread(target=get_global_agg, args=(octant_fea[:, batch_size:2 * batch_size], octant_fea_shape, agg_2))
    threads.append(t2)
    t3 = threading.Thread(target=get_global_agg, args=(octant_fea[:, 2 * batch_size:3 * batch_size], octant_fea_shape, agg_3))
    threads.append(t3)
    t4 = threading.Thread(target=get_global_agg, args=(octant_fea[:, 3 * batch_size:4 * batch_size], octant_fea_shape, agg_4))
    threads.append(t4)
    t5 = threading.Thread(target=get_global_agg, args=(octant_fea[:, 4 * batch_size:5 * batch_size], octant_fea_shape, agg_5))
    threads.append(t5)
    t6 = threading.Thread(target=get_global_agg, args=(octant_fea[:, 5 * batch_size:6 * batch_size], octant_fea_shape, agg_6))
    threads.append(t6)
    t7 = threading.Thread(target=get_global_agg, args=(octant_fea[:, 6 * batch_size:7 * batch_size], octant_fea_shape, agg_7))
    threads.append(t7)
    t8 = threading.Thread(target=get_global_agg, args=(octant_fea[:, 7 * batch_size:], octant_fea_shape, agg_8))
    threads.append(t8)

    for t in threads:
        t.setDaemon(False)
        t.start()
    for t in threads:
        if t.isAlive():
            t.join()
    global_agg_final = agg_1 + agg_2 + agg_3 + agg_4 + agg_5 + agg_6 + agg_7 + agg_8
    global_agg_final = np.concatenate(global_agg_final, axis=0)

    global_agg_final = np.transpose(np.array(global_agg_final), [1, 0, 2])  # (50, 33, 32)

    # print("global_agg_final:", global_agg_final.shape)

    return global_agg_final


def global_agg(octant_fea):
    final_feature_agg = []
    for i in range(len(octant_fea)):  # (4, Ln, B, N, dim)
        # print("train:", i)
        one_hop_node = np.array(np.transpose(np.squeeze(octant_fea[i]), [1, 2, 0]))
        # print("one_hop_node:", one_hop_node.shape)  # (B, N, dim)
        agg_temp = global_agg_multi(one_hop_node)
        # print("agg_temp:", agg_temp.shape)
        agg_temp = agg_temp.reshape(agg_temp.shape[0], -1)
        final_feature_agg.append(agg_temp)

    return final_feature_agg


def six_cones_agg(pc_data, octant_fea, alpha):
    final_feature_agg = []
    for i in range(len(octant_fea)):  # (4, Ln, B, N, 1)
        # print("train:", i)
        cones_idx = get_six_cones_idx(pc_data[i], alpha)

        one_hop_node = np.array(np.transpose(np.squeeze(octant_fea[i]), [1, 2, 0]))
        # print("one_hop_node:", one_hop_node.shape)  # (B, N, dim)

        six_agg_temp = get_six_cones_aggs_multi(cones_idx, one_hop_node, pc_data[i])

        # print("train_FPFH_hist_1", six_agg_temp.shape)  # (6, 300, 24, 32)
        final_feature_agg.append(six_agg_temp)

    return final_feature_agg


def xyz2sph(xyz):
    '''

    Args:
        xyz:

    Returns: x,y,z,r,elevation,azimuth

    '''
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)  # r
    ptsnew[:, 4] = np.arctan2(xyz[:, 2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
    ptsnew[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0])  # azimuth
    return ptsnew


def get_six_cones_idx(data, alpha):
    '''

    Args:
        data: (B, N, 3)

    Returns: True or False matrix (6, B, N)

    '''
    # Get Sph coordinates
    data_shape = data.shape
    data_sph = xyz2sph(data.reshape(-1, 3)).reshape(data_shape[0], data_shape[1], -1)  # (B, N, 6)
    # (x,y,z,r,elevation,azimuth)
    # print("train_data_sph:", data_sph)
    # print("train_data_sph:", data_sph.shape)

    # Get cone masks
    cones_idx = []
    cones_idx.append(data_sph[:, :, 4] >= (np.pi / 2 - alpha / 2))  # z axis up
    cones_idx.append(data_sph[:, :, 4] <= -(np.pi / 2 - alpha / 2))  # z axis down
    cones_idx.append((data_sph[:, :, 4] >= - alpha / 2) * (data_sph[:, :, 4] <= alpha / 2) *
                     (data_sph[:, :, 5] >= (np.pi / 2 - alpha / 2)) * (data_sph[:, :, 5] <= (np.pi / 2 + alpha / 2)) *
                     (np.cos(data_sph[:, :, 4]) * np.sin(data_sph[:, :, 5]) >= np.cos(alpha / 2)))  # y axis up
    cones_idx.append((data_sph[:, :, 4] >= - alpha / 2) * (data_sph[:, :, 4] <= alpha / 2) *
                     (data_sph[:, :, 5] >= (-np.pi / 2 - alpha / 2)) * (data_sph[:, :, 5] <= (-np.pi / 2 + alpha / 2)) *
                     (np.cos(data_sph[:, :, 4]) * np.sin(data_sph[:, :, 5]) <= -np.cos(alpha / 2)))  # y axis down
    cones_idx.append((data_sph[:, :, 4] >= - alpha / 2) * (data_sph[:, :, 4] <= alpha / 2) *
                     (data_sph[:, :, 5] >= - alpha / 2) * (data_sph[:, :, 5] <= alpha / 2) *
                     (np.cos(data_sph[:, :, 4]) * np.cos(data_sph[:, :, 5]) >= np.cos(alpha / 2)))  # x axis up
    cones_idx.append((data_sph[:, :, 4] >= - alpha / 2) * (data_sph[:, :, 4] <= alpha / 2) *
                     ((data_sph[:, :, 5] >= (np.pi - alpha / 2)) + (data_sph[:, :, 5] <= (-np.pi + alpha / 2))) *
                     (np.cos(data_sph[:, :, 4]) * np.cos(data_sph[:, :, 5]) <= -np.cos(alpha / 2)))  # x axis down

    cones_idx.append(cones_idx[0] + cones_idx[1])
    cones_idx.append(cones_idx[2] + cones_idx[3])
    cones_idx.append(cones_idx[4] + cones_idx[5])
    
    cones_idx.append(np.squeeze(inverted_cone_mask(data_sph[:, :, :3], np.array([1, 0, 0]))))
    cones_idx.append(np.squeeze(inverted_cone_mask(data_sph[:, :, :3], np.array([-1, 0, 0]))))
    cones_idx.append(np.squeeze(inverted_cone_mask(data_sph[:, :, :3], np.array([0, 1, 0]))))
    cones_idx.append(np.squeeze(inverted_cone_mask(data_sph[:, :, :3], np.array([0, -1, 0]))))
    cones_idx.append(np.squeeze(inverted_cone_mask(data_sph[:, :, :3], np.array([0, 0, 1]))))
    cones_idx.append(np.squeeze(inverted_cone_mask(data_sph[:, :, :3], np.array([0, 0, -1]))))

    cones_idx = cones_idx[6:]     
    # print("cones_idx:", np.array(cones_idx).shape)  # (6, B, N)

    return cones_idx


def inverted_cone_mask(point_data, x_tip):
    # Get inverted cone masks
    dir_axes = np.array([0, 0, 0]) - x_tip
    height = 1
    base_rad = 1
    cone_dist = np.dot((point_data - x_tip), dir_axes)
    cone_radius = (cone_dist / height) * base_rad
    dir_axes = dir_axes.reshape((1, -1))
    cone_dist = np.expand_dims(cone_dist, axis=-1)
    dum = np.squeeze(np.dot(cone_dist, dir_axes))
    orth_dist = np.linalg.norm(((point_data - x_tip) - dum), ord=2, axis=-1)
    cone_radius = np.expand_dims(cone_radius, axis=-1)
    orth_dist = np.expand_dims(orth_dist, axis=-1)
    mask_cone1 = (orth_dist < cone_radius) * (0 < cone_dist) * (cone_dist < height)

    return mask_cone1
    

def get_six_cones_aggs_multi(six_cones_idx, octant_fea, pc_data, n_jobs=-1):
    print("getting six cones aggregation")
    # print("octant_fea:", octant_fea.shape)
    # print("pc_data:", pc_data.shape)
    six_cones_agg_final = Parallel(n_jobs=n_jobs, verbose=1)(
        get_six_cones_hists([x[j] for x in six_cones_idx], octant_fea[j], pc_data[j]) for j in range(len(octant_fea)))

    six_cones_agg_final = np.concatenate(six_cones_agg_final, axis=0)

    # (33, 6, 50, 32)
    # print("train_hist:", six_cones_agg_final.shape) # (9840, 24, 9, 7)
    six_cones_agg_final = np.transpose(np.array(six_cones_agg_final), [2, 0, 3, 1])  # (6, 50, 33, 32)

    # print("six_cones_agg_final:", six_cones_agg_final.shape)

    return six_cones_agg_final


@delayed
@wrap_non_picklable_objects
def get_six_cones_hists(six_cones_idx, octant_feature, pc_data):
    agg_six_cones = []
    for i in range(octant_feature.shape[-1]):
        octant_one_feature = octant_feature[:, i]  # (B, N, 1)
        agg_one_six_cones = []
        for k in range(len(six_cones_idx)):  # 6
            octant_one_feature_temp = np.expand_dims(octant_one_feature[np.squeeze(np.array(np.where(six_cones_idx[k])))], axis=-1)
            if octant_one_feature_temp.size==0:
                nearest_center_point_idx = np.argmin(np.linalg.norm(pc_data,
                                                                    ord=1, axis=-1, keepdims=False))
                octant_one_feature_temp = [octant_one_feature[nearest_center_point_idx]]
            agg_one_temp = []
            agg_one_temp.append(np.mean(octant_one_feature_temp))
            agg_one_temp.append(np.max(octant_one_feature_temp))
            agg_one_temp.append(np.min(octant_one_feature_temp))
            agg_one_temp.append(np.var(octant_one_feature_temp))
            agg_one_temp.append(np.std(octant_one_feature_temp))
            agg_one_temp.append(np.linalg.norm(octant_one_feature_temp, ord=1, keepdims=False))
            agg_one_temp.append(np.linalg.norm(octant_one_feature_temp, ord=2, keepdims=False))

            if len(np.array(agg_one_temp).shape) == 1:
                agg_one_six_cones.append(agg_one_temp)
            else:
                agg_one_six_cones.append(np.concatenate(agg_one_temp, axis=-1))

        agg_six_cones.append(agg_one_six_cones)

    agg_six_cones = [np.array(agg_six_cones)]

    return agg_six_cones