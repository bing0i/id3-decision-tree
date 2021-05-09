import math
import os


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


def startDecisionTree(path):
    unreadPaths = []
    chosenAttribute = []
    data = readInputFile(path)
    targetAttribute = data["columnNames"][-1]
    selectedAttributes = [targetAttribute]

    with open("result.csv", "w") as outputFile:
        outputFile.write("begin\n")
        while len(selectedAttributes) != len(data["columnNames"]):
            outputFile.write("\n".join([str(v) for v in data["lines"]]) + "\n")

            bestAttribute, informationGains = chooseBestAttribute(data, targetAttribute, selectedAttributes)
            selectedAttributes.append(bestAttribute)
            outputFile.write("Information Gain\n")
            outputFile.write(",".join([str(v) for v in informationGains.keys()]) + "\n")
            outputFile.write(",".join([str("{:.3f}".format(v)) for v in informationGains.values()]) + "\n")
            outputFile.write("best attribute," + bestAttribute + "\n")

            rowValues = countValuesByRow(data["columnAttributes"][bestAttribute], data["columnAttributes"][targetAttribute])
            newRowValues, leaveNodes = removeLeafNode(rowValues)
            if len(leaveNodes) == 0:
                outputFile.write("non-leaf node found\n")
            else:
                outputFile.write("leaf node found\n")
                for key, value in leaveNodes.items():
                    outputFile.write(bestAttribute + ":" + "".join([k for k in key]) + "," + "".join([v for v in value]) + "\n")

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
            chosenAttribute = chosenAttribute[1:]
        outputFile.write("finish")


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


if __name__ == "__main__":
    startDecisionTree("buycomputer.csv")
