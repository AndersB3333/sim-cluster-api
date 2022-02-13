from flask import Flask, request, json
from flask_cors import CORS, cross_origin

import numpy as np
import pandas as pd
import random as random
import math
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
@cross_origin()
def post():
    points_dict = { 
    1:	0, 2:	0, 3:	0, 4:	0, 5:	0, 6: 	0, 7:	0, 8:	0, 9:	0, 10:	0, 11:	0, 12:	0, 13:	0, 14:	0, 15:	0, 16:	0, 17: 0, 18:	0,
19:	0, 20:	0, 21:	0, 22:	0, 23:	0, 24:	0, 25:	0, 26:	0, 27:	0, 28:	1, 29:	0, 30:	0, 31:	0, 32:	0, 33:	0, 34:	0, 35:	0, 36:	0, 37:1,
38:	1, 39:	2, 40:	1, 41:	1, 42:	0, 43:	0, 44:	0, 45:	0, 46:	0, 47:	1, 48:	2, 49:	3, 50:	4, 51:	3, 52:	2, 53:	1, 54:	0, 55:	0,
56:	0, 57:	1, 58:	2, 59:	3, 60:	4, 61:	5, 62:	4, 63:	3, 64:	2, 65:	1, 66:	0, 67:	0, 68:	0, 69:	1, 70:	2, 71:	3, 72:	4, 73:	3, 74:2,
75:	1, 76:	0, 77:	0, 78:	0, 79:	0, 80:	0, 81:	1, 82:	1, 83:	2, 84: 1,  85:	1, 86:	0, 87:	0, 88:	0, 89:	0, 90:	0, 91:	0, 92:	0, 93:0,
94:	1, 95:	0, 96:	0, 97:	0, 98:	0, 99:	0, 100:	0, 101:	0, 102:	0, 103:	0, 104:	0, 105:	0, 106:	0, 107:	0, 108:	0, 109:	0, 110:	0, 111:	0, 112:	0, 113:	0, 114:	0, 115:	0, 116:	0, 117:	0, 118:	0, 119:	0, 120:	0, 121:	0,
}
    request_data = request.get_json()
    df = pd.DataFrame(request_data, columns=['value'])
    pref_hand = df.value.iloc[-1]
    df.drop(df.tail(1).index, inplace=True)
    coordinates = []
    for y in range(5, -6, -1):
        for x in range(-5, 6):
            coordinates.append([x,y])
    counted__cluster_values = df[df.value > 0]
    strong_bins_area = []
    for i in counted__cluster_values.index:
        strong_bins_area.append(i)
    empty_list = []
    for i in strong_bins_area:
        empty_list.append(coordinates[i])
    k_num = round(math.sqrt(len(empty_list)))
    kmeans = KMeans(n_clusters = k_num, init = 'k-means++')
    kmeans = kmeans.fit_predict(empty_list)
    coord_cluster_list = [[]] * len(strong_bins_area)
    for count, strong_bin in enumerate(strong_bins_area):
        coord_cluster_list[count] = [strong_bin]
    for count, cluster_num in enumerate(kmeans):
        coord_cluster_list[count].append(cluster_num)
    cluster_list = []
    placeholder_list = []
    for i in range(k_num):
        cluster_list += [[]]
        placeholder_list += [[]]
    np_coord_cluster_list = np.array(coord_cluster_list)
    for count, value in enumerate(np_coord_cluster_list[:, 1]):
        cluster_list[value].append(np_coord_cluster_list[count,0])
    index = 0
    for count, outer_value in enumerate(cluster_list):
        for inner_value in outer_value:
            placeholder_list[index] += [coordinates[inner_value]]
        index += 1
    def centroid_calc(given_list):
        centroid = []
        x_cor_sum = 0
        y_cor_sum = 0
        for count, cord in enumerate(given_list):
            x_cor_sum += given_list[count][0]
            y_cor_sum += given_list[count][1]
        x_ave = x_cor_sum / len(given_list)
        y_ave = y_cor_sum / len(given_list)
        centroid.append(x_ave)
        centroid.append(y_ave)
        return centroid
    cluster_centroids = []
    for count, outer_value in enumerate(placeholder_list):
        cluster_centroids.append(centroid_calc(outer_value))
    total_shots = df.value.sum()
    df['relative_freq']= round(df.value / total_shots,4)
    strong_bins = []
    values_list = []
    for i in df.value:
        values_list.append(i)
    test_points = []
    for count, value in enumerate(values_list):
        test_points.append(points_dict[count+1] * value )
    t_sum = 0
    for i in test_points:
        t_sum += i
    for i, val in enumerate(df.relative_freq):
        if val >= 0.1:
            strong_bins.append(i)
    if len(strong_bins) == 0:
        for i, val in enumerate(df.relative_freq):
            if val >= 0:
                strong_bins.append(i)
    strong_bins_cord = []
    for i in strong_bins:
        strong_bins_cord.append(coordinates[i])
    qual_score = t_sum / total_shots
    qual_score = abs((abs(qual_score - 6) / 7))
    adj_rel_list = []
    for i in df.relative_freq:
        adj_rel_list.append(i)
    total_count = 0
    for i in adj_rel_list:
        if i > 0:
            total_count +=1
    zero_bins = 121-total_count
    zero_spread = total_shots
    zero_bin_list = []
    for count, value in enumerate(adj_rel_list):
        if value == 0:
            zero_bin_list.append(count)
            adj_rel_list[count] = adj_rel_list[count] + (zero_spread / zero_bins)
    def cor_dist_calc(cor1,cor2):
        dist_x = (cor2[0]-cor1[0])
        dist_y =  (cor2[1]-cor1[1])
        list = [dist_x, dist_y]
        final = math.sqrt(list[0]**2 + list[1]**2)
        return final
    def cor_dir_calc(cor1,cor2):
        dir_x = (cor2[0] - cor1[0])
        dir_y = (cor2[1] - cor1[1])
        angle = math.degrees(math.atan2(dir_y, dir_x))
        return angle
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
    if pref_hand == 0:
        dir_function = cor_dir_prob_r
    else:
        dir_function = cor_dir_prob_l
    def prob_applier(centroid, i, value):
        distance = cor_dist_calc(centroid, i)
        direction = dir_function(cor_dir_calc(centroid, i))
        final = value * direction / (1+distance)**2 + abs(distance -15) * qual_score / 100
        final =  final * 1000
        return final
    for centroid in cluster_centroids:
        for count, value in enumerate(adj_rel_list):
            if coordinates[count] == centroid:
                if coordinates[count] in strong_bins_cord:
                    adj_rel_list[count] += ((value) * 0.25 / (1.8 **2) + (15 *qual_score/ 100)) * 1000
                else: adj_rel_list[count] += ((value) * 0.25 / (1.8 **2) + (15 * qual_score / 1000000)) * 1000
            elif coordinates[count] in strong_bins_cord:
                adj_rel_list[count] += prob_applier(centroid, coordinates[count], value) * .9
            else:
                adj_rel_list[count] += prob_applier(centroid, coordinates[count], value)
    tot_adj_sum = 0
    for i in adj_rel_list:
        tot_adj_sum += i
    tot_sum = 0
    for i in adj_rel_list:
        tot_sum +=i
    for i in zero_bin_list:
        if adj_rel_list[i] >= tot_sum * .05:
            adj_rel_list[i] = adj_rel_list[i] * .10
    cumulat_bins = []
    cumulat_sum = 0
    for i in adj_rel_list:
        cumulat_bins.append(cumulat_sum)
        cumulat_sum += i
    rand_freq_list = []
    for i in range(1000):
        rand = round(random.uniform(min(cumulat_bins), max(cumulat_bins)),5)
        rand_freq_list.append(rand)
        rand_freq_list.sort()
    index = 0
    cumulat_bins_count = [0]*121
    for count, value in enumerate(rand_freq_list):

        index = 0
        if cumulat_bins[119] < value <= cumulat_bins[120]:
            cumulat_bins_count[120] +=1
        while value > cumulat_bins[index +1] and index < 119:
            index +=1

        if value < cumulat_bins[index + 1]:
            cumulat_bins_count[index] += 1
        else:
            cumulat_bins_count[index-1] += 1
    cumulat_bins_count = json.jsonify(cumulat_bins_count)
    return cumulat_bins_count


if __name__ == '__main__':
    app.run(port=5000, debug=True)