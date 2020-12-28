from decisionTree import *
from matplotlib import pyplot as plt 
import numpy as np  
import sys
import csv
from inspection import inspectData
import copy
import math






def main():

    trainInput = sys.argv[1]
    testInput = sys.argv[2]
    #maxDepthTmp = sys.argv[3]
    
    """
    outputTrainFile = sys.argv[4]
    outputTestFile = sys.argv[5]
    outputMetrics = sys.argv[6]
    """


    tmpTrainData = csv.reader(open(trainInput), delimiter = "\t")
    trainDataTmp = np.array(list(tmpTrainData))

    tmpTestData = csv.reader(open(testInput), delimiter = "\t")
    testDataTmp = np.array(list(tmpTestData))

    attributesLstTrain = copy.deepcopy(trainDataTmp)[0, 0:-1]
    attributesLstTest = copy.deepcopy(testDataTmp)[0, 0:-1]

    trainData = np.reshape(trainDataTmp, (len(trainDataTmp), len(trainDataTmp[0])))
    testData = np.reshape(testDataTmp, (len(testDataTmp), len(testDataTmp[0])))
    
    currentDepth = 0
    #maxDepth = int(maxDepthTmp)
   
    trainDataCopy = copy.deepcopy(trainData)
    outputLst = trainDataCopy[1:,-1]

    class1 = outputLst[0]
    class2 = ""

    for row in outputLst:
        if row != class1:
            class2 = row

    classes = [class1, class2]
    

    """
    node = Node(None, None, None, None, None, 0, 0)

    tree = buildDecisionTree(trainData, attributesLstTrain, maxDepth, currentDepth, classes, node)

    predictedTrainLabels = predictLabels(trainData, maxDepth, attributesLstTrain, tree)
    predictedTestLabels = predictLabels(testData, maxDepth, attributesLstTest, tree)

    #writeLabelsFile(predictedTrainLabels, outputTrainFile, predictedTestLabels, outputTestFile)

    trainingError = trainError(trainData, predictedTrainLabels)
    testingError = testError(testData, predictedTestLabels)

    """

    errorTrainVals = []
    errorTestVals = []

    maxDepthValsTrain = []
    maxDepthValsTest = []


    for maxDepth in range(0, len(attributesLstTrain)):


        maxDepthValsTrain.append(maxDepth)

        node = Node(None, None, None, None, None, 0, 0)
        tree = buildDecisionTree(trainData, attributesLstTrain, maxDepth, currentDepth, classes, node)
        predictedTrainLabels = predictLabels(trainData, maxDepth, attributesLstTrain, tree)
        
        trainingError = trainError(trainData, predictedTrainLabels)
        errorTrainVals.append(trainingError)



    for maxDepth in range(0, len(attributesLstTest)):

        maxDepthValsTest.append(maxDepth)

        node = Node(None, None, None, None, None, 0, 0)
        tree = buildDecisionTree(trainData, attributesLstTrain, maxDepth, currentDepth, classes, node)

        predictedTestLabels = predictLabels(testData, maxDepth, attributesLstTest, tree)

        testingError = testError(testData, predictedTestLabels)
        errorTestVals.append(testingError)




    plt.figure()
    plt.suptitle("Training and Testing Error of Decision Trees of Varying Depths\n(Evaluated on Politicians Datasets)")
    plt.plot(maxDepthValsTrain, errorTrainVals, label = "Train Error")
    plt.plot(maxDepthValsTest, errorTestVals, label = "Test Error")

    plt.xlabel("Max Depth (<= numAttributes)")
    plt.ylabel("Train/Test Error")
    plt.legend(loc = "lower right")
    plt.show()









if __name__ == '__main__':
    main()