from sklearn.cluster import KMeans
import cv2
import numpy as np
from scipy.stats import norm



def group_list (list , min_dist):
    list = np.array(list)
    g_start_index = 0
    g_end_index = 1
    output = []
    for i in range(1 , len(list)):
        if(list[i] - list[i-1] > min_dist):
            output.append(int(np.mean(list[g_start_index:g_end_index])))
            g_start_index = i
        g_end_index += 1
    output.append(int(np.mean(list[g_start_index:g_end_index])))
    return np.array(output)

def cluster_distances (distances):
    distances=distances.reshape(-1,1)
    init = np.array([np.min(distances) ,np.max(distances)]).reshape(-1,1)
    k_means = KMeans(n_clusters=2 , init=init , n_init=1 , max_iter=2).fit(distances)
    return k_means.cluster_centers_

def extract_connected_components_features(contours , hierarchy):
    ######## These set of features are extracted based on This Paper : ##########
    ############ A Set of Novel Features for Writer Identification ##############

    #get the largest contour index (the contour that is bounding the whole image)
    contour_lengthes = np.array([contour.shape[0] for contour in contours])
    largest_contour_index = np.argsort(contour_lengthes)[-1]

    # line_img = color.gray2rgb(line)

    #get parent contours only and their bounding boxes
    bounding_boxes = []
    bounding_areas = []
    for i , contour in enumerate(contours):
        if hierarchy[0][i][3] == largest_contour_index:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append([x,y,w,h])
            bounding_areas.append(w*h)
    bounding_areas = np.array(bounding_areas)
    bounding_boxes = np.array(bounding_boxes)

    #remove small contours
    average_word_area = np.percentile( bounding_areas, 75)
    bounding_boxes = bounding_boxes[np.where(bounding_areas>average_word_area*0.2)[0]]
    bounding_boxes = bounding_boxes[np.argsort(bounding_boxes[:,0])]

    #calculate the distances between contours
    distances = []
    for i in range(1 , bounding_boxes.shape[0]):
        x1, y1, w1, h1 = bounding_boxes[i-1]
        x2, y2, w2, h2 = bounding_boxes[i]
        distances.append(x2-(x1+w1))
    distances = np.array(distances)

    clusters = cluster_distances(distances)
    average_distance = np.average(distances)
    midian_distance = np.median(distances)
    std = np.sqrt(np.var(distances))
    average_distance_within_word = clusters[0][0]
    average_distance_between_words = clusters[1][0]

    return  [average_distance , midian_distance , std , average_distance_within_word , average_distance_between_words]


def extract_base_line_features(image):
    ######## These set of features are extracted based on This Paper : ##########
    ########## Writer Identification Using Text Line Based Features #############

    line_inverted = 1 - (image / np.max(image))
    horizontal_projection = np.sum(line_inverted, axis=1)

    repeated_indecies = []
    for i, count in enumerate(horizontal_projection):
        for j in range(int(count)):
            repeated_indecies.append(i)

    mu, sigma = norm.fit(np.array(repeated_indecies))
    mu = int(mu)
    sigma = int(sigma)

    non_white_indecies = np.where(horizontal_projection != 0)[0]

    top_line = non_white_indecies[0]
    bottom_line = non_white_indecies[-1]
    upper_base_line = mu - sigma
    lower_base_line = mu + sigma
    features = []
    f1 = top_line-upper_base_line
    f2 = upper_base_line-lower_base_line
    f3 = lower_base_line-bottom_line

    features.append(f1)
    features.append(f2)
    features.append(f3)

    features.append(f1/f2)
    features.append(f1/f3)
    features.append(f2/f3)

    return features


def extract_slant_features (image):
    ######## These set of features are extracted based on This Paper : ##########
    ####### Writer Identification Using Edge-Based Directional Features #########


    #apply canny filter to get image edges
    edges = 255 - cv2.Canny(image, 0, 250)
    line_inverted = 1 - (edges / np.max(image))
    features = []

    #get black pixels
    y_mask, x_mask = np.where(line_inverted == 1)

    #get slant features
    features.append(np.sum(line_inverted[y_mask - 1, x_mask]))
    features.append(np.sum(line_inverted[y_mask - 2, x_mask]))

    features.append(np.sum(line_inverted[y_mask + 1, x_mask + 1]))
    features.append(np.sum(line_inverted[y_mask + 1, x_mask + 2]))
    features.append(np.sum(line_inverted[y_mask + 2, x_mask + 1]))
    features.append(np.sum(line_inverted[y_mask + 2, x_mask + 2]))

    features.append(np.sum(line_inverted[y_mask - 1, x_mask + 1]))
    features.append(np.sum(line_inverted[y_mask - 1, x_mask + 2]))
    features.append(np.sum(line_inverted[y_mask - 2, x_mask + 1]))
    features.append(np.sum(line_inverted[y_mask - 2, x_mask + 2]))

    return features/np.sum(features)


def extract_width_features(image , writing_hight):
    ######## These set of features are extracted based on This Paper : ##########
    ########## Writer Identification Using Text Line Based Features #############

    line_inverted = 1 - (image / np.max(image))

    #get the row where most of the black - white transaction occur
    line_inverted_rolled = np.roll(line_inverted, 1, axis=1)
    transitions = (line_inverted != line_inverted_rolled).astype('int')
    most_transitions_index = np.argmax(np.sum(transitions, axis=1))

    #get lengths of consequent white pixels in the most transaction row
    zero_indeces = np.where(line_inverted[most_transitions_index] == 0)[0]
    count = 1
    white_pixels = []
    for i in range(1, len(zero_indeces)):
        if (zero_indeces[i] - zero_indeces[i - 1] == 1):
            count += 1
        else:
            white_pixels.append(count)
            count = 1
    white_pixels = np.array(white_pixels)

    f1 = np.median(white_pixels)
    f2 = f1 / writing_hight

    return [f1 , f2]

def get_features(line):
    img2, contours, hierarchy = cv2.findContours(line, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    features = np.array([])
    base_line_features = extract_base_line_features(line)
    text_height = base_line_features[1]

    features = np.append(features , extract_connected_components_features(contours, hierarchy))
    features = np.append(features , base_line_features)
    features = np.append(features , extract_slant_features(line))
    features = np.append(features , extract_width_features(line , text_height) )

    return features


