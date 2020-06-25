import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage.measurements import label
from scipy import ndimage
from scipy import signal
from scipy import stats
from skimage.measure import regionprops


def gaussian_kernel(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def get_detections(prob_map, prob_threshold=0.95, max_filt=5):
    '''
    Threshold heatmap to extract detection locations
    :param prob_map: 2D probability map output by the network
    :param prob_threshold: (float) probability threshold
    :param max_filt: (int) max_filter radius
    :return: (x, y) coords of detection points as a np array
    '''
    seg = prob_map > prob_threshold
    detections_y, detections_x = np.where(
        np.multiply(seg, prob_map == ndimage.filters.maximum_filter(prob_map, max_filt)))
    detections = np.append(detections_y[:, None], detections_x[:, None], axis=1)
    return detections


def order_detections(heatmap, detections):
    '''
    Order detections in strength from min to max
    :param heatmap: 2D heatmap of probabilities
    :param detections: (x, y) detection array output by get_detections()
    :return: sorted (x, y) detection array
    '''
    detection_strength = heatmap[detections[:, 0], detections[:, 1]]
    detection_order = np.argsort(detection_strength)[::-1]  # flip detection order to min -> max
    sorted_strength = detection_strength[detection_order]  # re-order by strength
    sorted_detections = detections[detection_order, :]  # y, x points sorted by strength
    return sorted_detections


def filter_detections(sorted_detections, max_dist=5):
    '''
    Remove points which are too close together (double counts)
    :param sorted_detections: sorted (x, y) detection array
    :param max_dist: (int) maximum distance allowed between detections measured in pixels
    :return: ordered (x, y) detections with double counts filtered out
    '''
    det_dist = cdist(sorted_detections, sorted_detections)
    det_indices = np.arange(0, sorted_detections.shape[0])
    det_dist[det_indices, det_indices] = np.inf

    too_close_j, too_close_i = np.where(det_dist < max_dist)

    retain = too_close_j > too_close_i
    too_close_j = too_close_j[retain]
    too_close_i = too_close_i[retain]
    keep = np.ones((len(sorted_detections),), dtype=bool)

    for i, j in zip(too_close_i, too_close_j):
        if keep[i]:
            keep[j] = False

    filtered_detections = sorted_detections[keep, :]
    return filtered_detections


def get_true_pos(true_locations, filtered_detections, max_dist=3):
    '''
    Calculate the number of true positive matches by comparing ground truth to detections
    :param true_locations: (x, y) coords of albatrosses based on ground truth map
    :param filtered_detections: sorted and filtered (x, y) detections output by network
    :param max_dist: (int) tollerance for detection matching the ground truth location in pixels
    :return: (int) number of sure positive matches
    '''
    match_dist = cdist(true_locations, filtered_detections)
    match_true, match_det = linear_sum_assignment(match_dist)
    valid_matches = match_dist[match_true, match_det] < max_dist
    number_matches = valid_matches.sum()
    return number_matches


def get_true_locations(target_patch):
    '''
    Extract (x, y) coords of detections from the binary target array
    :param target_patch: binary numpy array showing albatrosses
    :return: (x, y) coords of albatross locations
    '''
    target_label, num_features = label(target_patch)
    target_props = regionprops(target_label)
    true_locations = [r.centroid for r in target_props]
    if len(true_locations) > 0:
        true_locations = np.array(true_locations)
    else:
        true_locations = np.zeros((0, 2))
    return true_locations


def get_matches(true_locations, filtered_detections, max_dist=3):
    '''
    Compare ground truth and network detections and calculate matches
    :param true_locations: (x, y) array of ground truth locations
    :param filtered_detections: (x, y) list of network predicted locations
    :param max_dist: (int) tolerance for correct detection measured in pixels
    :return: lists of true positive (TP), false negative (FN) and false positive (FP) coordinates
    '''
    # Identify the matches
    match_dist = cdist(true_locations, filtered_detections)
    match_true, match_det = linear_sum_assignment(match_dist)
    valid_matches = match_dist[match_true, match_det] < max_dist

    # Indices of matches and misses
    tp_ind = match_true[valid_matches]
    fn_ind = [i for i in np.arange(len(true_locations)) if i not in tp_ind]
    fp_ind = [i for i in np.arange(len(filtered_detections)) if i not in match_det[valid_matches]]

    TPs = true_locations[tp_ind, :].astype(int)
    FNs = true_locations[fn_ind, :].astype(int)
    FPs = filtered_detections[fp_ind, :].astype(int)

    return TPs, FNs, FPs


def calculate_recall_precision(predictions, target, threshold):
    '''
    Calculate recall and precision over whole test set for a specified probability threshold
    :param predictions: prediction array output by the network (n_patches, n_classes, height, width)
    :param target: binary ground truth array (n_patches, n_classes, height, width)
    :param threshold: (float) probability threshold
    :return: recall and precision value for the dataset
    '''
    n_patches = predictions.shape[0]

    true_pos = np.zeros(n_patches)
    false_pos = np.zeros(n_patches)
    false_neg = np.zeros(n_patches)
    true_neg = np.zeros(n_patches)

    for patch in range(n_patches):
        true_locations = get_true_locations(target[patch, 0, :, :])
        detections = get_detections(predictions[patch, 1, :, :], threshold, 5)
        sorted_detections = order_detections(predictions[patch, 1, :, :], detections)
        filtered_detections = filter_detections(sorted_detections, 5)

        tp = get_true_pos(true_locations, filtered_detections, 3)

        fp = len(filtered_detections) - tp
        fn = len(true_locations) - tp
        tn = predictions.shape[0] * predictions.shape[1] - fn - fp - tp

        true_pos[patch] = tp;
        false_pos[patch] = fp
        false_neg[patch] = fn;
        true_neg[patch] = tn

    print('Errors for threshold {:.3%} : TPs = {}, FPs = {}, FNs = {}'.format(threshold, true_pos.sum(),
                                                                              false_pos.sum(), false_neg.sum()))

    # Recall and precision
    recall = true_pos.sum() / (true_pos.sum() + false_neg.sum())
    precision = true_pos.sum() / (true_pos.sum() + false_pos.sum())
    print('TPs {} / FNs {} / FPs {}'.format(true_pos.sum(), false_neg.sum(), false_pos.sum()))
    print('Recall: {:.3%} / Precision: {:.3%}'.format(recall, precision))

    return recall, precision


def recall_precision_curve(predictions, target):
    '''
    Calculate recall and precision for a range of probability values
    :param predictions: prediction array output by the network (n_patches, n_classes, height, width)
    :param target: binary ground truth array (n_patches, n_classes, height, width)
    :return: lists containing recall and precision values for each probability
    '''
    recall = [];
    precision = []
    for threshold in np.linspace(0.05, 0.95, 19):
        try:
            rec, prec = calculate_recall_precision(predictions, target, threshold)
        except:
            rec = np.nan;
            prec = np.nan
        recall.append(rec);
        precision.append(prec)

    return recall, precision
