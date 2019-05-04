from Preprocessing import *
from DataExtraction import *
from Models import *
from FeatureExtraction import *
from sklearn.decomposition import PCA

from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.preprocessing import StandardScaler



############################# preparing dataSet  #####################3##############

#load object that was previously stored by "extractDataSetDiffDist" function in DataExtraction
# dataSet = extractDataSetDiffDist("AlldataLastVersion.txt" , areaOfInterest=True)
# writeObjToFile("dataSet.txt" , dataSet)
dataSet = loadObjFromFile("dataSet.txt")
xTrain,yTrain,xDev,yDev ,fTrain ,fDev = manageDataRatios(dataSet ,8,3)

testSet=constructTestSet('testDataWithDola.txt',areaOfInterest=True,isIncludeRotated=False,isDolaIncluded=True)
fTest=np.array(testSet['X'])
yTest=np.array(testSet['y'])

fTrainNewSpeeds=fTrain[:,:5]
fDevNewSpeeds=fDev[:,:5]
fTestNewSpeeds=fTest[:,:5]

# normalizing the features
yDev=yDev.ravel()
SS=StandardScaler()
fTrainScaled = SS.fit_transform(fTrainNewSpeeds)
fDevScaled = SS.transform(fDevNewSpeeds)
fTestScaled=SS.transform(fTestNewSpeeds)

fTrainScaled=np.hstack((fTrainScaled[:,0:2].reshape(-1,2),fTrainScaled[:,2:3]))
fDevScaled= np.hstack((fDevScaled[:,0:2].reshape(-1,2),fDevScaled[:,2:3]))



############################# trying different models #####################3##############

# the model that was finally used NN with 1 hidden layer with 2 hidden units
preds,probas,coefs_,intercepts = trainNN(fTrainScaled , yTrain.ravel() , fDevScaled , yDev.ravel() , (2) , randomState=110,alpha=0,solver='lbfgs')
confusion_mat, precision, recall, fpr, fnr , acc , f1Score = EvaluateTheAccuracy(yDev, preds,"NN")
predsNN=preds

preds = trainSVM(fTrainScaled , yTrain.ravel() , fDevScaled , yDev.ravel() ,w=2)
confusion_mat, precision, recall, fpr, fnr , acc , f1Score = EvaluateTheAccuracy(yDev, preds,"SVM")
predsSVM=preds

preds = trainRF(fTrainScaled , yTrain.ravel() , fDevScaled , yDev.ravel())
confusion_mat, precision, recall, fpr, fnr , acc , f1Score = EvaluateTheAccuracy(yDev, preds,"Random Forest")
predsRF=preds

preds = trainKnn(fTrainScaled , yTrain.ravel() , fDevScaled , yDev.ravel(),9)
confusion_mat, precision, recall, fpr, fnr , acc , f1Score = EvaluateTheAccuracy(yDev, preds,"Knn")
predsKnn=preds

preds = trainLR(fTrainScaled , yTrain.ravel() , fDevScaled , yDev.ravel(),1)
confusion_mat, precision, recall, fpr, fnr , acc , f1Score = EvaluateTheAccuracy(yDev, preds,"Logi Regr.")
predsLR=preds

preds = trainLinSVM(fTrainScaled , yTrain.ravel() , fDevScaled , yDev.ravel(),2,w=3)
confusion_mat, precision, recall, fpr, fnr , acc , f1Score = EvaluateTheAccuracy(yDev, preds,"Lin SVM")
predsLinSVM=preds

