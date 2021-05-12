import math
import os
import ast
import argparse

def readInputFile(path):
    data = {
        "columnNames": [],
        "columnAttributes": {},
        "rowAttributes": [],
    }
    with open(path, "r") as inputFile:
        data["lines"] = [line.rstrip("\n") for line in inputFile]

        data["columnNames"] = data["lines"][0].split(",")

        for line in data["lines"][1:]:
            data["rowAttributes"].append(line.split(","))

        for index, attribute in enumerate(data["columnNames"]):
            data["columnAttributes"][attribute] = [
                line.split(",")[index] for line in data["lines"][1:]
            ]

    return data


def countValuesByColumn(attribute):
    resultValues = {}
    for value in attribute:
        if value not in resultValues.keys():
            resultValues[value] = 1
        else:
            resultValues[value] += 1

    return resultValues


def computeEntropy(attribute):
    values = countValuesByColumn(attribute)
    entropy = 0
    for frequency in values.values():
        entropy += -frequency / len(attribute) * math.log(frequency / len(attribute), 2)

    return entropy


def countValuesByRow(attribute, targetAttribute):
    resultValues = {}
    for index, targetValue in enumerate(targetAttribute):
        if attribute[index] not in resultValues.keys():
            resultValues[attribute[index]] = {targetValue: 1}
        else:
            if targetValue not in resultValues[attribute[index]]:
                resultValues[attribute[index]][targetValue] = 1
            else:
                resultValues[attribute[index]][targetValue] += 1

    return resultValues


def computeAverageEntropy(attribute, targetAttribute):
    tmpEntropy = 0
    averageEntropy = 0
    frequencyByTarget = countValuesByRow(attribute, targetAttribute)
    frequency = countValuesByColumn(attribute)

    for value, targetValues in frequencyByTarget.items():
        for fre in targetValues.values():
            tmpEntropy += -fre / frequency[value] * math.log(fre / frequency[value], 2)
        tmpEntropy *= frequency[value] / len(targetAttribute)
        averageEntropy += tmpEntropy
        tmpEntropy = 0

    return averageEntropy


def computeInformationGain(entropy, averageEntropy):
    return entropy - averageEntropy


def chooseBestAttribute(data, targetAttribute, selectedAttributes):
    maxColumnName = ""
    maxInformationGain = 0
    informationGains = {}

    entropy = computeEntropy(data["columnAttributes"][targetAttribute])
    for attribute in data["columnNames"]:
        if attribute in selectedAttributes:
            continue

        averageEntropy = computeAverageEntropy(data["columnAttributes"][attribute], data["columnAttributes"][targetAttribute])
        informationGain = computeInformationGain(entropy, averageEntropy)

        if informationGain > maxInformationGain:
            maxColumnName = attribute
            maxInformationGain = informationGain

        informationGains[attribute] = informationGain

    return maxColumnName, informationGains


def startDecisionTree(path, logPath):
    unreadPaths = []
    chosenAttribute = []
    data = readInputFile(path)
    targetAttribute = data["columnNames"][-1]
    selectedAttributes = [targetAttribute]
    decisionTree = {}
    parentAttribute = ""
    parentValue = ""

    with open(logPath, "w") as outputFile:
        outputFile.write("begin\n")
        while len(selectedAttributes) != len(data["columnNames"]):
            outputFile.write("\n".join([str(v) for v in data["lines"]]) + "\n")

            bestAttribute, informationGains = chooseBestAttribute(data, targetAttribute, selectedAttributes)
            selectedAttributes.append(bestAttribute)
            outputFile.write("Information Gain\n")
            outputFile.write(",".join([str(v) for v in informationGains.keys()]) + "\n")
            outputFile.write(",".join([str("{:.3f}".format(v)) for v in informationGains.values()]) + "\n")
            outputFile.write("best attribute," + bestAttribute + "\n")

            if parentAttribute == "":
                decisionTree[bestAttribute] = {"parent": None, "valueOfParent": None, "children": []}
            elif bestAttribute not in decisionTree.keys():
                decisionTree[bestAttribute] = {"parent": parentAttribute, "valueOfParent": parentValue, "children": []}

            rowValues = countValuesByRow(data["columnAttributes"][bestAttribute], data["columnAttributes"][targetAttribute])
            newRowValues, leaveNodes = removeLeafNode(rowValues)
            if len(leaveNodes) == 0:
                outputFile.write("non-leaf node found\n")
            else:
                outputFile.write("leaf node found\n")
                for key, value in leaveNodes.items():
                    outputFile.write(bestAttribute + ":" + "".join([k for k in key]) + "," + "".join([v for v in value]) + "\n")
                    decisionTree[bestAttribute]["children"].append({data["columnNames"][-1]: {"parent": bestAttribute, "valueOfParent": key, "value": "".join([v for v in value])}})
            
            if (len(newRowValues) == 0 and len(unreadPaths) == 0):
                break

            unreadPaths += writeNewDatasetToTempFile(data, newRowValues, bestAttribute)
            if (len(unreadPaths) != 0):
                data = readInputFile(unreadPaths[0])
                os.remove(unreadPaths[0])
                unreadPaths = unreadPaths[1:]

            for key in newRowValues.keys():
                chosenAttribute.append("\n" + bestAttribute + ":" + key + "\n")
                    

            outputFile.write(chosenAttribute[0])

            indexOfColon = chosenAttribute[0].index(":")
            parentAttribute = chosenAttribute[0][1:indexOfColon]
            parentValue = chosenAttribute[0][indexOfColon + 1 : len(chosenAttribute[0]) - 1]

            chosenAttribute = chosenAttribute[1:]

        outputFile.write("finish")

    return decisionTree


def removeLeafNode(rowValues):
    newRowValues = {}
    removedNodes = {}
    for key, value in rowValues.items():
        if len(value) != 1:
            newRowValues[key] = value
        else:
            removedNodes[key] = value

    return newRowValues, removedNodes


def writeNewDatasetToTempFile(data, rowValues, bestAttribute):
    unreadPaths = []
    for index, value in enumerate(rowValues.keys()):
        tmpDataSet = []
        for line in data["rowAttributes"]:
            if line[data["columnNames"].index(bestAttribute)] == value:
                tmpDataSet.append(line)

        with open("tmpDataset" + str(index) + ".csv", "w") as outputFile:
            unreadPaths.append("tmpDataset" + str(index) + ".csv")
            outputFile.write(",".join([str(v) for v in data["columnNames"]]))
            outputFile.write("\n")

            lines = ""
            for line in tmpDataSet:
                lines += ",".join([str(word) for word in line]) + "\n"
            outputFile.write(lines)

    return unreadPaths


def writeModelFile(path, decisionTree):
    with open(path, "w") as modelFile:
        for attribute, values in decisionTree.items():
            modelFile.write("node\n")
            modelFile.write(str(attribute) + "\n")
            for key, value in values.items():
                modelFile.write(str(value) + "\n")

    return True


def readModelFile(path):
    with open(path, "r") as modelFile:
        lines = [line.rstrip('\n') for line in modelFile]
        
    model = {}
    isNode = False
    currentNode = ""
    values = {
        0: "parent",
        1: "valueOfParent",
        2: "children",
    }
    counter = 0

    for line in lines:
        if line == "node":
            isNode = True
            counter = 0
            continue
        elif isNode:
            isNode = False
            currentNode = line
            model[currentNode] = {"parent": None, "valueOfParent": None, "children": []}
        else:
            if counter == 2:
                model[currentNode][values[counter]] = ast.literal_eval(line)
            elif line == "None":
                model[currentNode][values[counter]] = None
            else:
                model[currentNode][values[counter]] = line
            counter += 1

    return model


def getTargetValue(model, values, attributes, index):
    if list(model.keys())[0] == attributes[-1]:
        if values[index] == model[attributes[-1]]["valueOfParent"]:
            return model[attributes[-1]]["value"]
        else:
            return None

    if attributes[index] not in model:
        for i in range(index + 1, len(attributes)):
            if attributes[i] in model:
                if model[attributes[i]]["valueOfParent"] == values[index]:
                    if "children" in model[attributes[i]]:
                        for child in model[attributes[i]]["children"]:
                            resultValue = getTargetValue(child, values, attributes, i + 1)
                            if resultValue != None:
                                return resultValue
                    elif "value" in model[attributes[i]]:
                        return model[attributes[i]]["value"]

    if model[attributes[index]]["valueOfParent"] == values[index]:
        if "children" in model[attributes[index]]:
            for child in model[attributes[index]]["children"]:
                resultValue = getTargetValue(child, values, attributes, index + 1)
                if resultValue != None:
                    return resultValue
            
            return getTargetValue(model, values, attributes, index + 1)


def predictTargetValue(inputPath, outputPath, model):
    data = {
        "columnNames": [],
        "rowAttributes": []
    }

    with open(inputPath, "r") as inputFile:
        lines = [line.rstrip("\n") for line in inputFile]
        data["columnNames"] = lines[0].split(",")
        for line in lines[1:]:
            data["rowAttributes"].append([None] + line.split(","))

    with open(outputPath, "w") as outputFile:
        outputFile.write(",".join([v for v in data["columnNames"]]) + "\n")
        for row in data["rowAttributes"]:
            outputFile.write(",".join([str(v) for v in row[1:]]) + "," + getTargetValue(model, row, data["columnNames"], 0) + "\n")

    return None


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='train or predict')
    parser.add_argument('--data', help='data file path')
    parser.add_argument('--model', help='model file path')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--result', help='result file path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArguments()

    if args.task == "train":
        decisionTree = startDecisionTree(args.data, args.log)
        writeModelFile(args.model, decisionTree)
    elif args.task == "predict":
        model = readModelFile(args.model)
        predictTargetValue(args.data, args.result, model)


