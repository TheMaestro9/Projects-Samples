import numpy as np
import random
from FileHandler import *
from Preprocessing import *
from FeatureExtraction import *

def compineData(x_matab, y_matab, x_ghalat, y_ghalat, shuffle=True):
    print(x_matab.shape, y_matab.shape, x_ghalat.shape, y_ghalat.shape)
    y_matab = y_matab.reshape(y_matab.shape[0], 1)
    y_ghalat = y_ghalat.reshape(y_ghalat.shape[0], 1)
    x = np.vstack((x_matab, x_ghalat))
    y = np.vstack((y_matab, y_ghalat))
    if (shuffle):
        perm = np.random.permutation(len(x))
        x = x[perm, :]
        y = y[perm, :]
    return x, y


# extract data set such that training and dev set come from different destribution
def extractDataSetDiffDist(fileName, areaOfInterest=False, includeDola=False):
    rows = loadObjFromFile(fileName)
    xTrainM = []
    yTrainM = []
    xDevM = []
    yDevM = []
    xTrainG = []
    yTrainG = []
    xDevG = []
    yDevG = []

    random.seed(1)
    random.shuffle(rows)
    index = 0
    for row in rows:

        if (index % 100 == 0):
            print("finished:", index, "rows")
        index += 1
        #         if(index > 100 ):
        #             break
        anamoly = Anamoly(JsonObj=row["value"])

        if (len(anamoly.accelTime) < 400):
            continue

        ######################## preprocessing ##########################
        # if we want area of interest we will normalize
        anamoly = preprossing(anamoly, smoothing=True, padding=True)
        interestSpeed = getInterestSpeed(anamoly, 2, 50)

        ####################### adding extra features ###################
        avgAbs = avgAbsRatio(anamoly, 2)
        areaOfInterestAnamoly = preprossing(anamoly, areaOfInterest=True, interestPeriod=2)
        peakCount = getNumberOfPeaks(areaOfInterestAnamoly.accelValues)
        zeroCrossing = zeroCrossings1D(np.array(areaOfInterestAnamoly.accelValues))

        anamoly = preprossing(anamoly, areaOfInterest=areaOfInterest, normalizing=areaOfInterest)
        ######### elemenate anamolies with missing accel Values #########

        #         print(len(anamoly.accelValues))
        if (areaOfInterest):
            expectedAccelSamples = 151
        else:
            expectedAccelSamples = 500
        if (len(anamoly.accelValues) != expectedAccelSamples):
            continue

        if (len(anamoly.speedValues) < 2):
            continue

        speedMean = np.mean(anamoly.speedValues)
        speedVar = np.var(anamoly.speedValues)

        anamoly.accelValues = np.append(anamoly.accelValues,
                                        [avgAbs, peakCount, zeroCrossing, interestSpeed, speedMean, speedVar])
        #         print(anamoly.accelValues.shape

        ######################  Training Set  ##########################

        if anamoly.anamolyType == 0 and "Reviewed" in row['value']:
            xTrainM.append(anamoly.accelValues)
            yTrainM.append(1)

        if (anamoly.anamolyType > 0 and anamoly.anamolyType < 4 and "test" not in anamoly.id):
            xTrainG.append(anamoly.accelValues)
            yTrainG.append(0)

        ######################## Dev Set ################################
        check = False
        if includeDola:
            check = (anamoly.anamolyType == 0 and "Adel" in anamoly.id and "Reviewed" not in row['value'])

        if (anamoly.anamolyType == 0 and "test" in anamoly.id) or check:
            xDevM.append(anamoly.accelValues)
            yDevM.append(1)

        if (anamoly.anamolyType > 0 and anamoly.anamolyType < 4 and "test" in anamoly.id):
            xDevG.append(anamoly.accelValues)
            yDevG.append(0)

            #     print(len(xTrainG))
    dataSet = {}
    dataSet['xTrainG'] = np.asarray(xTrainG)
    dataSet['xTrainM'] = np.asarray(xTrainM)
    dataSet['yTrainM'] = np.asarray(yTrainM)
    dataSet['yTrainG'] = np.asarray(yTrainG)
    dataSet['xDevG'] = np.asarray(xDevG)
    dataSet['yDevG'] = np.asarray(yDevG)
    dataSet['xDevM'] = np.asarray(xDevM)
    dataSet['yDevM'] = np.asarray(yDevM)

    return dataSet


# input:
# dataSet: dictionary containing all DataSet after prepocessing
# GtoMTrain: ghalat to matab Ratio for Training Set
# GtoMDev: ghalat to matab Ratio for Dev Set
# numberOfFeatures: number of extra features (not acceleration) that is appended to the acceleration values
#
# returns Training, Dev and extra Feature Sets
def manageDataRatios(dataSet, GtoMTrain, GtoMDev, numberOfFeatures=6):
    xTrainG = dataSet['xTrainG']
    xTrainM = dataSet['xTrainM']
    yTrainM = dataSet['yTrainM']
    yTrainG = dataSet['yTrainG']
    xDevG = dataSet['xDevG']
    yDevG = dataSet['yDevG']
    xDevM = dataSet['xDevM']
    yDevM = dataSet['yDevM']
    np.random.seed(0)
    trainPerm = len(xTrainM) * GtoMTrain
    devPerm = len(xDevM) * GtoMDev
    if (trainPerm < xTrainG.shape[0]):
        xTrainG = xTrainG[:trainPerm]
        yTrainG = yTrainG[:trainPerm]
    if (devPerm < xDevG.shape[0]):
        xDevG = xDevG[:devPerm]
        yDevG = yDevG[:devPerm]

    xTrain, yTrain = compineData(xTrainM, yTrainM, xTrainG, yTrainG)
    xDev, yDev = compineData(xDevM, yDevM, xDevG, yDevG)

    return xTrain[:, :-numberOfFeatures], yTrain, xDev[:, :-numberOfFeatures], yDev, xTrain[:, -numberOfFeatures:], xDev[:,-numberOfFeatures:]


def constructTestSet(fileName, areaOfInterest=False, isIncludeRotated=False, isDolaIncluded=False):
    rows = loadObjFromFile(fileName)

    random.seed(1)
    random.shuffle(rows)
    index = 0
    xTest = []
    yTest = []
    for row in rows:
        if not isDolaIncluded and (not isIncludeRotated and (row['value']['MLpred'] != row['value']['cosSimPred'])):
            continue

        if (index % 100 == 0):
            print("finished:", index, "rows")
        index += 1
        #         if(index > 100 ):
        #             break
        anamoly = Anamoly(JsonObj=row["value"])

        if (len(anamoly.accelTime) < 400):
            continue

        ######################## preprocessing ##########################
        # if we want area of interest we will normalize
        anamoly = preprossing(anamoly, smoothing=True, padding=True)
        interestSpeed = getInterestSpeed(anamoly, 2, 50)

        ####################### adding extra features ###################
        avgAbs = avgAbsRatio(anamoly, 2)
        areaOfInterestAnamoly = preprossing(anamoly, areaOfInterest=True, interestPeriod=2)
        peakCount = getNumberOfPeaks(areaOfInterestAnamoly.accelValues)
        zeroCrossing = zeroCrossings1D(np.array(areaOfInterestAnamoly.accelValues))

        anamoly = preprossing(anamoly, areaOfInterest=areaOfInterest, normalizing=areaOfInterest)
        ######### elemenate anamolies with missing accel Values #########

        #         print(len(anamoly.accelValues))
        if (areaOfInterest):
            expectedAccelSamples = 151
        else:
            expectedAccelSamples = 500
        if (len(anamoly.accelValues) != expectedAccelSamples):
            continue

        if (len(anamoly.speedValues) < 2):
            continue

        speedMean = np.mean(anamoly.speedValues)
        speedVar = np.var(anamoly.speedValues)

        anamoly.accelValues = [avgAbs, peakCount, zeroCrossing, interestSpeed, speedMean, speedVar]
        #         print(anamoly.accelValues.shape)

        if anamoly.anamolyType == 0:
            xTest.append(anamoly.accelValues)
            yTest.append(1)

        if anamoly.anamolyType > 0 and anamoly.anamolyType < 4:
            xTest.append(anamoly.accelValues)
            yTest.append(0)

    dataset = {}
    dataset['X'] = xTest
    dataset['y'] = yTest
    return dataset

