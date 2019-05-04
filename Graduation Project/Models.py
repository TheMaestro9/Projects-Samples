from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


def getKnnWithNewData(xTrain, yTrain, xTest, yTest, k):
    yTrain = yTrain.ravel()
    yTest = yTest.ravel()
    knr = KNeighborsClassifier(n_neighbors=k, n_jobs=1)
    print("[KNN] Training")
    knr.fit(xTrain, yTrain)
    preds = knr.predict(xTest)
    preds = np.array(preds).reshape(len(preds), 1)
    dists, neighbours = knr.kneighbors(xTest, k, True)
    print('get Score')
    acc = knr.score(xTest, yTest)
    print(acc)
    #   print(datetime.datetime.now())
    return acc, preds, neighbours, dists


def trainNN(xTrain, yTrain, xTest, yTest, layerSizes, randomState=100, alpha=0.1, solver='adam'):
    model = MLPClassifier(hidden_layer_sizes=layerSizes, random_state=randomState, max_iter=200000, alpha=alpha,
                          solver=solver)
    model.fit(xTrain, yTrain)
    trainingAcc = model.score(xTrain, yTrain)
    testAcc = model.score(xTest, yTest)
    print("NN training Accuracy: ", trainingAcc)
    print("NN test Accuracy: ", testAcc)
    return model.predict(xTest), model.predict_proba(xTest), model.coefs_, model.intercepts_


def trainRF(xTrain, yTrain, xTest, yTest):
    model = RandomForestClassifier(random_state=100)
    model.fit(xTrain, yTrain)
    trainingAcc = model.score(xTrain, yTrain)
    testAcc = model.score(xTest, yTest)
    print("Random Forest training Accuracy: ", trainingAcc)
    print("Random Forest test Accuracy: ", testAcc)
    return model.predict(xTest)


def trainSVM(xTrain, yTrain, xTest, yTest, C=1, w=1):
    model = SVC(random_state=100, C=C, class_weight={1: w})
    model.fit(xTrain, yTrain)
    trainingAcc = model.score(xTrain, yTrain)
    testAcc = model.score(xTest, yTest)
    print("SVM training Accuracy: ", trainingAcc)
    print("SVM test Accuracy: ", testAcc)
    return model.predict(xTest)


def trainKnn(xTrain, yTrain, xTest, yTest, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xTrain, yTrain)
    trainingAcc = model.score(xTrain, yTrain)
    testAcc = model.score(xTest, yTest)
    print("Knn training Accuracy: ", trainingAcc)
    print("Knn test Accuracy: ", testAcc)
    return model.predict(xTest)


def trainLR(xTrain, yTrain, xTest, yTest, n=2):
    poly = PolynomialFeatures(n)
    xTrain = poly.fit_transform(xTrain)
    xTest = poly.transform(xTest)
    SS = StandardScaler()
    xTrain = SS.fit_transform(xTrain)
    xTest = SS.transform(xTest)
    model = LogisticRegression(random_state=100)
    model.fit(xTrain, yTrain)
    trainingAcc = model.score(xTrain, yTrain)
    testAcc = model.score(xTest, yTest)
    print("Logi Regr. training Accuracy: ", trainingAcc)
    print("Logi Regr. Accuracy: ", testAcc)
    print("Logi Regr. learned weights:")
    params_dict = {}
    feat_names = poly.get_feature_names()
    coeffs = np.squeeze(model.coef_)
    for i in range(len(feat_names)):
        params_dict[feat_names[i]] = coeffs[i]
    print(params_dict)
    return model.predict(xTest)


def trainLinSVM(xTrain, yTrain, xTest, yTest, n=2, w=1):
    poly = PolynomialFeatures(n)
    xTrain = poly.fit_transform(xTrain)
    xTest = poly.transform(xTest)
    model = SVC(random_state=100, kernel='linear', class_weight={1: w})
    model.fit(xTrain, yTrain)
    trainingAcc = model.score(xTrain, yTrain)
    testAcc = model.score(xTest, yTest)
    print("Lin. SVM training Accuracy: ", trainingAcc)
    print("Lin. SVM Accuracy: ", testAcc)
    return model.predict(xTest)


def EvaluateTheAccuracy(y_true, yPredicted, modelName=""):
    y_true = y_true.ravel()
    yPredicted = yPredicted.ravel()
    confusion_mat = confusion_matrix(y_true, yPredicted)
    tn, fp, fn, tp = confusion_matrix(y_true, yPredicted).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    acc = np.sum(y_true == yPredicted) / y_true.shape[0]
    f1Score = 2 * precision * recall / (precision + recall)
    print(modelName, "confusion matrix: ", confusion_mat)
    print(modelName, "precision: ", precision)
    print(modelName, "recall: ", recall)
    print(modelName, "accuracy: ", acc)
    print(modelName, "F1 Score: ", f1Score)
    print('-----------------------------------------------')

    return confusion_mat, precision, recall, fpr, fnr, acc, f1Score


