import os
import numpy as np
import re
from segmentation import do_segmentation
from features import get_features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pickle
from sklearn.preprocessing import StandardScaler

main_path = "testing_data/test_cases"
number = re.compile("[0-9]*")

def rearrange_dir(elements):
    numbers = []
    others =[]
    for element in elements:
        if(number.fullmatch(element)):
            numbers.append(element)
        else :
            others.append(element)
    return np.append(np.sort(np.array(numbers).astype("int")).astype("str") , np.array(others))

#function that extract feature form image given its path
def process_image(image_path, writer=-1):
    image_lines = do_segmentation(image_path)
    x_train = []
    y_train = []
    for line in image_lines:
        x_train.append(get_features(line))
        y_train.append(writer)
    return np.array(x_train) , np.array(y_train).reshape(-1,1)

#function that extract features from test image and predicts its class based on the pretrained model
def predict_test_label(trained_model , test_image_path= None , test_features =None):
    if(test_image_path != None):
        x_test , _ = process_image(test_image_path)
    else:
        x_test= test_features
    predictions = trained_model.predict(x_test)
    (values, counts) = np.unique(predictions, return_counts=True)
    return values[np.argmax(counts)]

def extract_features(test_cases_path):
    all_test_case_data = []
    test_case_number = 1
    main_path = test_cases_path
    #for each test case
    for test_case in rearrange_dir(os.listdir(main_path)):
        test_case_path = main_path + "/" + test_case
        x_train = np.array([])
        y_train = np.array([])
        #for each known writer
        for writer in rearrange_dir(os.listdir(test_case_path)):
            writer_path = test_case_path + "/" + writer
            if (number.fullmatch(writer)):  # not the test image
                #for each image of writer's writing
                for image in rearrange_dir(os.listdir(writer_path)):
                    image_path = writer_path + "/" + image
                    #extract features
                    image_x_train, image_y_train = process_image(image_path, int(writer))
                    if (x_train.size == 0):
                        x_train = image_x_train
                        y_train = image_y_train
                    else:
                        x_train = np.vstack((x_train, image_x_train))
                        y_train = np.vstack((y_train, image_y_train))
        #extract features of test image
        test_img_path = test_case_path + "/" + "test.jpg"
        x_test, _ = process_image(test_img_path)
        all_test_case_data.append((x_train, y_train, x_test))
        print("Extracted Features for test case number: ", test_case_number)
        test_case_number += 1
    return all_test_case_data

def predict_writers(all_test_case_data , used_features):
    predictions = []
    test_case_number = 1
    # model = SVC(gamma=0.001)
    model = KNeighborsClassifier(n_neighbors=5)
    # model = MLPClassifier(hidden_layer_sizes=(10),max_iter=10000)
    for test_case in all_test_case_data:
        #get data of interest (may drop some of the features to see how they affect the accuracy)
        x_train , y_train , x_test = test_case
        x_train = x_train[: , used_features]
        x_test = x_test[:, used_features]

        #standerdize the data
        scaler = StandardScaler().fit(x_train)
        x_train= scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        #train the model and make predection for the test image
        model.fit(x_train , y_train)
        predictions.append(predict_test_label(model,test_features=x_test))
        print("finished Prediction for test case number" , test_case_number)
        test_case_number+=1

    return predictions

def calc_acc (predictions , true_writers_file_path):
    file = open(true_writers_file_path, 'r')
    true_writers = np.array(file.readlines()).astype("int")
    predictions = np.array(predictions)
    acc = np.sum(true_writers == predictions)/predictions.size *100
    return acc

def generate_used_feautres_vector(connected_components= False , base_line=False , slant=False , width = False  , all_features=False):

    CONNECTED_COMPONENTS = 5
    BASE_LINE_FEATURES = 6
    SLANT_FEATURES = 10
    WIDTH_FEATURES = 2
    ALL_FEAUTRES = CONNECTED_COMPONENTS + BASE_LINE_FEATURES +SLANT_FEATURES +WIDTH_FEATURES

    if(all_features):
        return np.ones((ALL_FEAUTRES ,)).astype('bool')
    used_features = np.zeros((ALL_FEAUTRES,))

    index = 0

    if(connected_components == True):
        used_features[index:index+CONNECTED_COMPONENTS] = 1
    index += CONNECTED_COMPONENTS

    if(base_line ==True):
        used_features[index:index+BASE_LINE_FEATURES] = 1
    index+= BASE_LINE_FEATURES

    if (slant == True):
        used_features[index:index + SLANT_FEATURES] = 1
    index += SLANT_FEATURES

    if (width == True):
        used_features[index:index +WIDTH_FEATURES] = 1
    index += WIDTH_FEATURES

    return used_features.astype("bool")

############################# uncomment this code to extract features #####################
#############################   form images and save them in file     #####################

# all_test_case_data = extract_features(main_path)
#
# output_file = open("all_test_case_data" , 'wb')
# pickle.dump(all_test_case_data , output_file)
# output_file.close()

############################# Read pre extracted Feature file #####################

input_file = open("all_test_case_data" , 'rb')
all_test_case_data = pickle.load(input_file)

############################# Classify And print Accuracy #####################
# print(len(all_test_case_data))
used_features = generate_used_feautres_vector(slant=True , connected_components=True)
predictions = predict_writers(all_test_case_data , used_features)

print("The Model ACC:" , calc_acc(predictions , "testing_data/true_writers.txt") , "%")
