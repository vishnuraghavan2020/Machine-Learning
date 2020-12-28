import math
import csv
import sys
import numpy as np 
import copy




def inspectData(trainData, classes):

    #print(data)
    trainDataCopy = copy.deepcopy(trainData)
    dataClass = trainDataCopy[1:, -1]



    #print(dataClass)
    class1Count = 0
    class2Count = 0

    class1 = dataClass[0]
    class2 = ""
    total = len(dataClass)

    for c in dataClass:
        if c != class1:
            class2 = c
            class2Count += 1
        else:
            class1Count += 1

    
    if class2 == "":
        for cl in classes:
            if cl != class1:
                class2 = cl

    
    pClass1 = class1Count / total
    pClass2 = class2Count / total

    tmpCond1 = 0
    tmpCond2 = 0
    if pClass1 == 0:
        tmpCond1 = 0
    else:
        tmpCond1 = pClass1*math.log(pClass1, 2)

    if pClass2 == 0:
        tmpCond2 = 0
    else:
        tmpCond2 = pClass2*math.log(pClass2, 2)


    entropy = -(tmpCond1 + tmpCond2)
    errorRate = min(class1Count, class2Count) / (class1Count + class2Count)

    return [entropy, errorRate, class1, class2, class1Count, class2Count]


def writeError(outputFile, data):

    entropy = data[0]
    errorRate = data[1]
    inspectFile = open(outputFile, "w")
    inspectFile.write("entropy: %f\n" % (entropy))
    inspectFile.write("error: %f" % (errorRate))
    



def main():

    inputFile = sys.argv[1]
    outputFile = sys.argv[2]

    print("The input file is: %s" % (inputFile))
    print("The output file is: %s" % (outputFile))

    tmpData = csv.reader(open(inputFile), delimiter = "\t")
    trainData = np.array(list(tmpData))

    trainDataCopy = copy.deepcopy(trainData)
    outputLst = trainDataCopy[1:,-1]

    class1 = outputLst[0]
    class2 = ""

    for row in outputLst:
        if row != class1:
            class2 = row

    classes = [class1, class2]
    
    data = inspectData(trainData, classes)
    writeError(outputFile, data)





if __name__ == '__main__':
    main()