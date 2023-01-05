import math
import random

import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, request, json
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)


@app.route('/', methods=['POST'])
@cross_origin()
def post():
    points_dict = {
        1:	0, 2:	0, 3:	0, 4:	0, 5:	0, 6: 	0, 7:	0, 8:	0, 9:	0,
        10:	0, 11:	0, 12:	0, 13:	0, 14:	0, 15:	0, 16:	0, 17: 0, 18:	0,
        19:	0, 20:	0, 21:	0, 22:	0, 23:	0, 24:	0, 25:	0, 26:	0, 27:	0,
        28:	1, 29:	0, 30:	0, 31:	0, 32:	0, 33:	0, 34:	0, 35:	0, 36:	0,
        37: 1, 38:	1, 39:	2, 40:	1, 41:	1, 42:	0, 43:	0, 44:	0, 45:	0,
        46:	0, 47:	1, 48:	2, 49:	3, 50:	4, 51:	3, 52:	2, 53:	1, 54:	0,
        55:	0, 56:	0, 57:	1, 58:	2, 59:	3, 60:	4, 61:	5, 62:	4, 63:	3,
        64:	2, 65:	1, 66:	0, 67:	0, 68:	0, 69:	1, 70:	2, 71:	3, 72:	4,
        73:	3, 74: 2,  75:	1, 76:	0, 77:	0, 78:	0, 79:	0, 80:	0, 81:	1,
        82:	1, 83:	2, 84: 1,  85:	1, 86:	0, 87:	0, 88:	0, 89:	0, 90:	0,
        91:	0, 92:	0, 93: 0,  94:	1, 95:	0, 96:	0, 97:	0, 98:	0, 99:	0,
        100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106:	0, 107:	0, 108:	0,
        109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115:	0, 116:	0, 117:	0,
        118: 0, 119: 0, 120: 0, 121: 0,
    }
    request_data = request.get_json()

    # Storing the values as a numpy array for faster processing
    arr = np.array(request_data)
    # Declaring whether the player was right (0), or left(1) handed, and
    # how many shots to be simulated
    PREF_HAND, SHOTS_SIM = arr[-2:]
    # Deleting the values last two values since they're stored as separate
    # variables
    arr = np.delete(arr, [-2, -1])
    # Counting the values the user has shot
    counted__cluster_values = arr[arr > 0]
    strong_bins_area = np.where(arr > 0)[0]
    # creating a cartesian coordinate system to make it easier to calculate
    # the L2-norm later on
    coordinates = list()
    for y in range(5, -6, -1):
        for x in range(-5, 6):
            coordinates.append([x, y])
    coordinates = np.array(coordinates)
    # Converting the strong bins coordinates to a list for kMeans section
    # TODO: change this line to coordinates[strong_bins_area]
    list_convert_coord_strong_bins = [[j for j in i]
                                      for i in coordinates[strong_bins_area]]
    # Declaring the number of clusters.
    k_num = round(math.sqrt(len(coordinates[strong_bins_area])))
    kmeans = KMeans(n_clusters=k_num, init='k-means++')
    # Fitting the clusters based on the frequency of the shots in the
    # different bins
    kmeans = kmeans.fit_predict(list_convert_coord_strong_bins)
    # Converting the bins with highest frequency, i.e. >0, to their
    # cartesian coordinates
    coord_cluster_list = [[]] * len(strong_bins_area)
    for count, strong_bin in enumerate(strong_bins_area):
        coord_cluster_list[count] = [strong_bin]
    for count, cluster_num in enumerate(kmeans):
        coord_cluster_list[count].append(cluster_num)
    # Creating two empy lists, cluster list is storing the index number of
    # the clusters, whereas placeholder list is storing the cartesian
    # distance from the center, referred here as "centroid"
    placeholder_list = [[] for _ in range(k_num)]
    cluster_list = [[] for _ in range(k_num)]

    np_coord_cluster_list = np.array(coord_cluster_list)
    for count, value in enumerate(np_coord_cluster_list[:, 1]):
        cluster_list[value].append(np_coord_cluster_list[count, 0])
    for count, outer_value in enumerate(cluster_list):
        for inner_value in outer_value:
            placeholder_list[count] += [list(coordinates[inner_value])]

    # Calculates the center point, "centroid", of the bins that is > 0

    def centroid_calc(given_list):
        centroid = []
        x_cor_sum = 0
        y_cor_sum = 0
        for count in range(len(given_list)):
            x_cor_sum += given_list[count][0]
            y_cor_sum += given_list[count][1]
        x_ave = x_cor_sum / len(given_list)
        y_ave = y_cor_sum / len(given_list)
        centroid.append(x_ave)
        centroid.append(y_ave)
        return centroid

    cluster_centroids = list()
    for count, outer_value in enumerate(placeholder_list):
        cluster_centroids.append(centroid_calc(outer_value))

    # Finding the level of the player (to determine higher probability
    # to the target, and bins close to it)
    # Determining the total number of shots the user has entered
    total_shots = sum(arr)
    # Finding the relative frequency of where these shots occured
    relative_freq = np.round(arr / sum(arr), decimals=4)

    values_list = list()
    for i in arr:
        values_list.append(i)
    test_points = []
    for count, value in enumerate(arr):
        test_points.append(points_dict[count+1] * value)
    t_sum = sum(test_points)
    strong_bins = list(np.where(relative_freq > .1)[0])

    # Making sure strong bins is not zero, this could happen if the
    # player's shot distribution is wide, or if someone is testing this
    # program
    if len(strong_bins) == 0:
        for i, val in enumerate(relative_freq):
            if val >= 0:
                strong_bins.append(i)

    strong_bins_cord = coordinates[strong_bins]
    # storing the "quality" of the player with the qual_score variable
    qual_score = t_sum / total_shots
    # Reversing the quality of the player for the probability algorithm
    qual_score = abs((abs(qual_score - 6) / 7))
    adj_rel_list = np.copy(relative_freq)
    np_adj_rel_list = np.array(relative_freq)
    total_count = len(np_adj_rel_list[np_adj_rel_list > 0])
    zero_bins = 121 - total_count
    # Applying some probability to the values that are currently zero, so
    # the program will not multiply by zero
    zero_spread = total_shots * .1
    zero_bin_list = []
    for count, value in enumerate(relative_freq):
        if value == 0:
            zero_bin_list.append(count)
            relative_freq[count] = relative_freq[count] + \
                (zero_spread / zero_bins)

    # Determining the euclidean distance between the two coordinates
    # the formula being used is the regular Frobenius norm
    def cor_dist_calc(cor1, cor2):
        return np.linalg.norm(cor2 - cor1)

    # Finding the direction between the two coordinates
    def cor_dir_calc(cor1, cor2):
        dir_x = (cor2[0] - cor1[0])
        dir_y = (cor2[1] - cor1[1])
        angle = math.degrees(math.atan2(dir_y, dir_x))
        return angle

    # Assigning the probability for right handed players
    def cor_dir_prob_r(angle):
        if -1 <= angle <= 1:
            return 0.18
        elif 1 < angle < 20:
            return 0.165
        elif 20 <= angle < 40:
            return 0.15
        elif 40 <= angle < 50:
            return 0.12
        elif 50 <= angle < 70:
            return 0.15
        elif 70 <= angle < 85:
            return 0.165
        elif 85 <= angle < 95:
            return 0.18
        elif 95 <= angle < 115:
            return 0.2
        elif 115 <= angle < 125:
            return 0.22
        elif 125 <= angle < 140:
            return 0.25
        elif 140 <= angle < 155:
            return 0.22
        elif 155 <= angle < 175:
            return 0.2
        elif 175 <= angle <= 180:
            return 0.18
        elif -179 <= angle < -175:
            return 0.18
        elif -175 <= angle < -165:
            return 0.165
        elif -165 <= angle < -140:
            return 0.15
        elif -140 <= angle < -130:
            return 0.12
        elif -130 <= angle < -110:
            return 0.15
        elif -110 <= angle < -95:
            return 0.165
        elif -95 <= angle < -85:
            return 0.18
        elif -85 <= angle < -65:
            return 0.2
        elif -65 <= angle < -50:
            return 0.22
        elif -50 <= angle < -40:
            return 0.25
        elif -40 <= angle < -20:
            return 0.22
        elif -20 <= angle <= -1:
            return 0.2
        else:
            raise ValueError("Unsupported value: {}".format(angle))

    # Probability of the direction of left-handed players
    def cor_dir_prob_l(angle):
        if -1 <= angle <= 1:
            return 0.18
        elif 1 < angle < 20:
            return 0.2
        elif 20 <= angle < 40:
            return 0.22
        elif 40 <= angle < 50:
            return 0.25
        elif 50 <= angle < 70:
            return 0.22
        elif 70 <= angle < 85:
            return 0.2
        elif 85 <= angle < 95:
            return 0.18
        elif 95 <= angle < 115:
            return 0.165
        elif 115 <= angle < 125:
            return 0.15
        elif 125 <= angle < 140:
            return 0.12
        elif 140 <= angle < 155:
            return 0.15
        elif 155 <= angle < 175:
            return 0.165
        elif 175 <= angle <= 180:
            return 0.18
        elif -179 <= angle < -175:
            return 0.18
        elif -179 <= angle < -165:
            return 0.2
        elif -165 <= angle < -140:
            return 0.22
        elif -140 <= angle < -130:
            return 0.25
        elif -130 <= angle < -110:
            return 0.22
        elif -110 <= angle < -95:
            return 0.2
        elif -95 <= angle < -85:
            return 0.18
        elif -85 <= angle < -65:
            return 0.165
        elif -65 <= angle < -50:
            return 0.15
        elif -50 <= angle < -40:
            return 0.12
        elif -40 <= angle < -20:
            return 0.15
        elif -20 <= angle <= -1:
            return 0.165
        else:
            raise ValueError("Unsupported value: {}".format(angle))

    # Declaring the dictionary to be used based on the players hitting
    # direction
    if PREF_HAND == 0:
        dir_function = cor_dir_prob_r
    else:
        dir_function = cor_dir_prob_l

    # The probability applier function, the values were arbitrarily
    # created to make it as realistic as possible
    def prob_applier(centroid, i, value):
        distance = cor_dist_calc(centroid, i)
        direction = dir_function(cor_dir_calc(centroid, i))
        final = value * direction / (1+distance)**2 + \
            abs(distance - 15) * qual_score / 100000
        final = final * 1000
        return final

    # Looping through the centroids to assign probabilities to each bin
    for centroid in cluster_centroids:
        for count, value in enumerate(adj_rel_list):
            if np.array_equal(coordinates[count], np.array(centroid)):
                if coordinates[count] in np.array(strong_bins_cord):
                    adj_rel_list[count] += ((value) * 0.25
                                            / (1.9 ** 2) + (15 * qual_score / 100000)) * 1000
                else:
                    adj_rel_list[count] += ((value) * 0.25
                                            / (1.9 ** 2) + (15 * qual_score / 100000)) * 1000
            elif coordinates[count] in np.array(strong_bins_cord):
                adj_rel_list[count] += prob_applier(centroid,
                                                    coordinates[count], value) * 0.47
            else:
                adj_rel_list[count] += prob_applier(centroid,
                                                    coordinates[count], value)

    # Searching to see if some values that was originally zero had increased
    # by more than 5% of the total value. If this is the case, the value will be
    # decreased by 10%
    tot_sum = sum(adj_rel_list)
    zero_bin_adj_rel_list = adj_rel_list[zero_bin_list]
    zero_bin_adj_rel_list = zero_bin_adj_rel_list[zero_bin_adj_rel_list >=
                                                  tot_sum * 0.05] * .1

    # Creating the cumulatative
    cumulat_bins = []
    cumulat_sum = 0
    for i in adj_rel_list:
        cumulat_bins.append(cumulat_sum)
        cumulat_sum += i
    rand_freq_list = []
    for i in range(SHOTS_SIM):
        rand = round(random.uniform(min(cumulat_bins), max(cumulat_bins)), 5)
        rand_freq_list.append(rand)
    rand_freq_list.sort()
    TOTAL_BINS = 121
    index = 0
    cumulat_bins_count = [0] * TOTAL_BINS
    for count, value in enumerate(rand_freq_list):
        index = 0
        if cumulat_bins[119] < value <= cumulat_bins[120]:
            cumulat_bins_count[120] += 1
        while value > cumulat_bins[index + 1] and index < 119:
            index += 1

        if value < cumulat_bins[index + 1]:
            cumulat_bins_count[index] += 1
        else:
            cumulat_bins_count[index-1] += 1

    json_bin_count = json.jsonify(cumulat_bins_count)
    return json_bin_count


if __name__ == '__main__':
    app.run(port=5000, debug=True)
