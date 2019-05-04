from Preprocessing import *

def zeroCrossings(X):
    XCrossings = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        x = X[i, :]
        for j in range(1, X.shape[1]):
            if (x[j] * x[j - 1]) < 0:
                XCrossings[i] += 1

    return XCrossings


def zeroCrossings1D(x):
    XCrossings = 0
    for j in range(1, len(x)):
        if (x[j] * x[j - 1]) < 0:
            XCrossings += 1

    return XCrossings


def normalize(anamoly):
    newAnamoly = Anamoly(anamoly=anamoly)
    absAccels = [abs(number) for number in newAnamoly.accelValues]
    maxx = max(absAccels)
    newAnamoly.accelValues = [x / maxx for x in newAnamoly.accelValues]
    return newAnamoly


def avgAbsRatioNew(anamoly, interestPeriod):
    _, start, end = getAreaOfInterest(anamoly, interestPeriod)
    avgAbsInterest = np.sum(np.abs(anamoly.accelValues[start:end])) / len(anamoly.accelValues[start:end])
    avgAbsNotInterest = np.mean(np.append(anamoly.accelValues[:start], anamoly.accelValues[:end]))
    return avgAbsInterest / avgAbsNotInterest


def avgAbsRatio(anamoly, interestPeriod):
    _, start, end = getAreaOfInterest(anamoly, interestPeriod)
    avgAbsInterest = np.sum(np.abs(anamoly.accelValues[start:end])) / len(anamoly.accelValues[start:end])
    avgAbsTotal = np.sum(np.abs(anamoly.accelValues)) / len(anamoly.accelValues)
    return avgAbsInterest / avgAbsTotal


def getNumberOfPeaks(values):
    peakCount = 0
    for i in range(1, len(values) - 1):
        if ((values[i] < values[i - 1] and values[i] < values[i + 1]) or (
                values[i] > values[i - 1] and values[i] > values[i + 1])):
            peakCount += 1
    return peakCount


def getMaxZCDist(x):
    maxDist = 0
    dist = 0
    for j in range(1, len(x)):
        dist += 1
        if (x[j] * x[j - 1]) < 0:
            maxDist = max(maxDist, dist)
            dist = 0
    return maxDist

