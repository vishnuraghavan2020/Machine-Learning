import numpy as np 
import sys
import csv
from inspection import inspectData
import copy
import math



def splitCol(trainData, attribute, attributesLst):

    trainDataCopy = copy.deepcopy(trainData)

    attCount = list(attributesLst).index(attribute)
    branchLeft = [list(trainDataCopy[0,:])]
    branchRight = [list(trainDataCopy[0,:])]
    
    val1 = trainDataCopy[1, attCount]
    val2 = ""
    
    for row in trainDataCopy[1:,:]:
        if row[attCount] != val1:
            val2 = row[attCount]
            branchRight.append(list(row))
            
        elif row[attCount] == val1:
            branchLeft.append(list(row))


    branchLeft = np.reshape(branchLeft, (len(branchLeft), len(branchLeft[0])))
    branchRight = np.reshape(branchRight, (len(branchRight), len(branchRight[0])))

    bundle = [branchLeft, branchRight, val1, val2]
    return bundle



def allZeroMutual(trainData, attributesLst, classes):
    
    for attribute in attributesLst:
        mI = mutualInfo(trainData, attribute, attributesLst, classes)
        if mI != 0:
            return False
    return True




def mutualInfo(trainData, attribute, attributesLst, classes):

    trainDataCopy = copy.deepcopy(trainData)

    entropy = inspectData(trainData, classes)[0]
    class1 = inspectData(trainData, classes)[2]
    class2 = inspectData(trainData, classes)[3]

    attCount = list(attributesLst).index(attribute)

    class1CountA = 0
    class2CountA = 0

    class1CountB = 0
    class2CountB = 0

    label1 = trainData[1, attCount]

    label2 = ""

    for row in trainDataCopy[1:, :]:
        if row[attCount] != label1:
            label2 = row[attCount]
            
    label1Count = 0
    label2Count = 0
    for row in trainDataCopy[1:, :]:
        if row[attCount] == label1 and row[-1] == class1:
            class1CountA += 1
            label1Count += 1
        elif row[attCount] == label1 and row[-1] == class2:
            class2CountA += 1
            label1Count += 1

        if row[attCount] == label2 and row[-1] == class1:
            class1CountB += 1
            label2Count += 1

        elif row[attCount] == label2 and row[-1] == class2:
            class2CountB += 1
            label2Count += 1

    if label1Count == 0 or label2Count == 0:
        return 0

    pClass1Label1 = class1CountA / label1Count 
    pClass2Label1 = class2CountA / label1Count 

    pClass1Label2 = class1CountB / label2Count 
    pClass2Label2 = class2CountB / label2Count 

    tmpCond1A = 0
    tmpCond1B = 0
    tmpCond2A = 0
    tmpCond2B = 0

    if pClass1Label1 == 0:
        tmpCond1A = 0
    else:
        tmpCond1A = pClass1Label1*math.log(pClass1Label1,2)

    if pClass2Label1 == 0:
        tmpCond1B = 0
    else:
        tmpCond1B = pClass2Label1*math.log(pClass2Label1,2)

    if pClass1Label2 == 0:
        tmpCond2A = 0
    else:
        tmpCond2A = pClass1Label2*math.log(pClass1Label2,2)

    if pClass2Label2 == 0:
        tmpCond2B = 0
    else:
        tmpCond2B = pClass2Label2*math.log(pClass2Label2,2)

    #H(Y | A = a)

    specCondEntropy1 = -(tmpCond1A + tmpCond1B)
    specCondEntropy2 = -(tmpCond2A + tmpCond2B)

    #H(Y | A)

    pA1 = label1Count / len(trainData[1:,:])
    pA2 = label2Count / len(trainData[1:,:])
    
    condEntropy = pA1*specCondEntropy1 + pA2*specCondEntropy2

    #I(Y ; A)

    mI = entropy - condEntropy
    return mI




def bestAttribute(trainData, attributesLst, classes):

    mutualDict = dict()
    for attribute in attributesLst:

        mI = mutualInfo(trainData, attribute, attributesLst, classes)
        mutualDict[attribute] = mI

    keyLst = []

    for key in mutualDict.keys():
        keyLst.append(key)

    bestVal = 0
    bestKey = keyLst[0]
    
    for key in mutualDict:
       
        if float(mutualDict[key]) > bestVal:
            bestVal = float(mutualDict[key])
            bestKey = key
        
    return bestKey
    



class Node(object):

    def __init__(self, att = None, val = "", left = None, right = None, label = "", countA = 0, countB = 0):
        self.att = att 
        self.val = ""
        self.left = left
        self.right = right
        self.label = ""
        self.countA = countA
        self.countB = countB



def majorityVote(class1, class2, class1Count, class2Count, node):

    
    node.countA = class1Count
    node.countB = class2Count

    if class1Count == class2Count:
        if class1 > class2:
            node.val = str(class1)
        elif class2 > class1:
            node.val = str(class2)

    elif class1Count > class2Count:
        node.val = str(class1)
        
    elif class2Count > class1Count:
        node.val = str(class2)
        
    node.left = None
    node.right = None
    
    return node




def getOutputStats(trainData, classes):

    class1 = inspectData(trainData, classes)[2]
    class2 = inspectData(trainData, classes)[3]
    class1Count = inspectData(trainData, classes)[4]
    class2Count = inspectData(trainData, classes)[5]

    return [class1,class2, class1Count, class2Count]




def buildDecisionTree(trainData, attributesLst, maxDepth, currentDepth, classes, node):

    
    if maxDepth == 0: #Just a leaf node, perform majority vote

        print((currentDepth * "|") + "[%d %s / %d %s]" % (getOutputStats(trainData, classes)[2],
                                                          getOutputStats(trainData, classes)[0], 
                                                          getOutputStats(trainData, classes)[3], 
                                                          getOutputStats(trainData, classes)[1]))

        return majorityVote(getOutputStats(trainData, classes)[0], 
                            getOutputStats(trainData, classes)[1], 
                            getOutputStats(trainData, classes)[2], 
                            getOutputStats(trainData,classes)[3], node)
       

    currentDepth += 1
    
    
    if allZeroMutual(trainData, attributesLst, classes):

        return majorityVote(getOutputStats(trainData, classes)[0], 
                            getOutputStats(trainData, classes)[1], 
                            getOutputStats(trainData, classes)[2], 
                            getOutputStats(trainData,classes)[3], node)
         

     
    if currentDepth == maxDepth:
        
        attribute = bestAttribute(trainData, attributesLst, classes)
        node.att = attribute
        node = decisionStump(trainData, attributesLst, node, classes)
        
        print((currentDepth * "|") + "%s = %s: [%d %s / %d %s]" %(attribute, node.left.label, node.left.countA, 
            getOutputStats(trainData, classes)[0], 
            node.left.countB, 
            getOutputStats(trainData, classes)[1]))

        print((currentDepth * "|") + "%s = %s: [%d %s / %d %s]" %(attribute, node.right.label, node.right.countA, 
            getOutputStats(trainData, classes)[0], 
            node.right.countB, 
            getOutputStats(trainData, classes)[1]))
        
        return node

        
    
     
    else:
        
        att = bestAttribute(trainData, attributesLst, classes)
        split = splitCol(trainData, att, attributesLst)
        
        node.att = att
   
        nextNodeLeft = Node(None, "", None, None, "", 0, 0)
        nextNodeLeft.label = split[2]

        nextNodeRight = Node(None, "", None, None, "", 0, 0)
        nextNodeRight.label = split[3]

        print((currentDepth * "|") + "%s = %s: [%d %s / %d %s]" %(att, split[2] ,getOutputStats(split[0], classes)[2], 
            getOutputStats(split[0], classes)[0], 
            getOutputStats(split[0], classes)[3], 
            getOutputStats(split[0], classes)[1]))
        
        
        node.left = buildDecisionTree(split[0], attributesLst, maxDepth, currentDepth, classes, nextNodeLeft)
       
        print((currentDepth * "|") + "%s = %s: [%d %s / %d %s]" %(att , split[3],getOutputStats(split[1], classes)[2], 
            getOutputStats(split[1], classes)[0], 
            getOutputStats(split[1], classes)[3], 
            getOutputStats(split[1], classes)[1]))
       
        
        
        node.right = buildDecisionTree(split[1], attributesLst, maxDepth, currentDepth, classes, nextNodeRight)
        
        return node


def decisionStump(trainData, attributesLst, node, classes):

    att = node.att

    attCount = list(attributesLst).index(att)

    split = splitCol(trainData, att, attributesLst)

    leftBranch = split[0] 
    rightBranch = split[1]

    class1Left = leftBranch[1,-1]
    class2Left = ""

    class1CountLeft = 0
    class2CountLeft = 0

    for row in leftBranch[1:,:]:
        if row[-1] != class1Left:
            class2Left = row[-1]
            class2CountLeft += 1
        else:
            class1CountLeft += 1

    
    if class2Left == "":
        for cl in classes:
            if cl != class1Left:
                class2Left = cl
    

    class1Right = rightBranch[1,-1]
    class2Right = ""

    class1CountRight = 0
    class2CountRight = 0


    for row in rightBranch[1:,:]:
        if row[-1] != class1Right:
            class2Right = row[-1]
            class2CountRight += 1
        else:
            class1CountRight += 1

    
    if class2Right == "":
        for cl in classes:
            if cl != class1Right:
                class2Right = cl
    
    nextNodeLeft = Node(None, "", None, None, "", 0, 0)
    nextNodeLeft.label = split[2]

    nextNodeRight = Node(None, "", None, None, "", 0, 0)
    nextNodeRight.label = split[3]


    node.left = majorityVote(class1Left, class2Left, class1CountLeft, class2CountLeft, nextNodeLeft)
    node.right = majorityVote(class1Right, class2Right, class1CountRight, class2CountRight, nextNodeRight)

    return node 







def trainError(trainData, predictedTrainLabels):

    trainDataCopy = copy.deepcopy(trainData)
    actualLabels = trainDataCopy[1:,-1]
    errorCount = 0
    
    for idx in range(0, len(predictedTrainLabels)):

        if predictedTrainLabels[idx] != actualLabels[idx]:
            errorCount += 1

    trainError = float(errorCount) / len(predictedTrainLabels)
    return trainError




def testError(testData, predictedTestLabels):

    testDataCopy = copy.deepcopy(testData)
    actualLabels = testDataCopy[1:,-1]
    errorCount = 0
    
    for idx in range(0, len(predictedTestLabels)):

        if predictedTestLabels[idx] != actualLabels[idx]:
            errorCount += 1

    testError = float(errorCount) / len(predictedTestLabels)
    return testError



def writeMetrics(trainError, testError, outputMetricFile):

    outputMetric = open(outputMetricFile, "w")
    

    outputMetric.write("error(train): %f\n" % (trainError))
    outputMetric.write("error(test): %f" % (testError))

    



def predictLabels(trainData, maxDepth, attributesLst, decisionTree):

    trainDataCopyTmp = copy.deepcopy(trainData)
    trainDataCopy = trainDataCopyTmp[:,0:-1]
    predictedLabels = []
    
    if maxDepth == 0:
        predictedLabels.append(decisionTree.val)
        return predictedLabels 

    root = decisionTree.att
    idx = list(attributesLst).index(root)
    

    for row in trainDataCopy[1:,:]:
    
        if decisionTree.left.label == row[idx]:
            createPath(row, attributesLst, decisionTree.left, predictedLabels)
           
        elif decisionTree.right.label == row[idx]:
            createPath(row, attributesLst, decisionTree.right, predictedLabels)

    return predictedLabels
        


def createPath(data, attributesLst, decisionTree, predictedLabels):

    
    if decisionTree.left == None and decisionTree.right == None:
        
        predictedLabels.append(decisionTree.val)
        return predictedLabels 

    currAttribute = decisionTree.att
    attIdx = list(attributesLst).index(currAttribute)


    if decisionTree.left.label == data[attIdx]:
        return createPath(data, attributesLst, decisionTree.left, predictedLabels)

    elif decisionTree.right.label == data[attIdx]:
        return createPath(data, attributesLst, decisionTree.right, predictedLabels)


def writeLabelsFile(predictedTrainLabels, outputTrainFile, predictedTestLabels, outputTestFile):

    trainLabels = open(outputTrainFile, "w")

    for trainLabel in predictedTrainLabels:
        
        trainLabels.write(trainLabel)
        trainLabels.write("\n")


    testLabels = open(outputTestFile, "w")

    for testLabel in predictedTestLabels:

        testLabels.write(testLabel)
        testLabels.write("\n")





def main():

    trainInput = sys.argv[1]
    testInput = sys.argv[2]
    maxDepthTmp = sys.argv[3]
    outputTrainFile = sys.argv[4]
    outputTestFile = sys.argv[5]
    outputMetrics = sys.argv[6]

    print("The train input file is: %s\n" % (trainInput))
    
    tmpTrainData = csv.reader(open(trainInput), delimiter = "\t")
    trainDataTmp = np.array(list(tmpTrainData))

    tmpTestData = csv.reader(open(testInput), delimiter = "\t")
    testDataTmp = np.array(list(tmpTestData))

    attributesLstTrain = copy.deepcopy(trainDataTmp)[0, 0:-1]
    attributesLstTest = copy.deepcopy(testDataTmp)[0, 0:-1]

    trainData = np.reshape(trainDataTmp, (len(trainDataTmp), len(trainDataTmp[0])))
    testData = np.reshape(testDataTmp, (len(testDataTmp), len(testDataTmp[0])))
    
    currentDepth = 0
    maxDepth = int(maxDepthTmp)
   
    trainDataCopy = copy.deepcopy(trainData)
    outputLst = trainDataCopy[1:,-1]

    class1 = outputLst[0]
    class2 = ""

    for row in outputLst:
        if row != class1:
            class2 = row

    classes = [class1, class2]
    

    node = Node(None, None, None, None, None, 0, 0)

    print((currentDepth * "|") + "[%d %s / %d %s]" % (getOutputStats(trainData, classes)[2],
                                                          getOutputStats(trainData, classes)[0], 
                                                          getOutputStats(trainData, classes)[3], 
                                                          getOutputStats(trainData, classes)[1]))

    tree = buildDecisionTree(trainData, attributesLstTrain, maxDepth, currentDepth, classes, node)

    predictedTrainLabels = predictLabels(trainData, maxDepth, attributesLstTrain, tree)
    predictedTestLabels = predictLabels(testData, maxDepth, attributesLstTest, tree)

    #writeLabelsFile(predictedTrainLabels, outputTrainFile, predictedTestLabels, outputTestFile)

    trainingError = trainError(trainData, predictedTrainLabels)
    testingError = testError(testData, predictedTestLabels)

    #writeMetrics(trainingError, testingError, outputMetrics)


    
if __name__ == '__main__':
    main()