
# A basic NN with 3 layers of 2-3-1 nodes

import random as rand
import math


def init():

    global learningRate
    global neur1, neur2, neur3, weights1, weights2, biases1, biases2, costs1, costs2

    learningRate = 0.001

    # Init Weights and Biases to random num from (-0.5,0.5)
    neur1 = [0,0]
    neur2 = [0,0,0]
    weights1 = []
    for i in range(0,len(neur2)):
        weights1 += [[]]
        for j in range(0,len(neur1)):
            weights1[i] += [rand.random()-0.5]
    biases1 = []
    for i in range(0,len(neur2)):
        biases1 += [rand.random()-0.5]
    costs1 = neur2

    neur3 = [0]
    weights2 = []
    for i in range(0,len(neur3)):
        weights2 += [[]]
        for j in range(0,len(neur2)):
            weights2[i] += [rand.random()-0.5]
    biases2 = []
    for i in range(0,len(neur3)):
        biases2 += [rand.random()-0.5]
    costs2 = neur3

    print("Set variables!")

    # Read training point data from Training Data.txt
    # Format - each line is a new data point of the form [Input 1],[Input 2],[Wanted Output]
    global data
    trainingPoints = 0
    data = []
    dataFile = open("Training Data.txt","r")
    if(dataFile == None):
        raise 
    dataPoint = []

    dataFile.seek(0)

    while(True):
        t = dataFile.readline()

        if(t == ""):
            break
        
        txt = []
        for i in range(0,len(t)):
            txt += [t[i]]

        txt.pop(-1)

        t = ""
        while(True):
            if(txt[0] == ","):
                txt.pop(0)
                break
            t += txt.pop(0)
        d1 = float(t)

        t = ""
        while(True):
            if(txt[0] == ","):
                txt.pop(0)
                break
            t += txt.pop(0)
        d2 = float(t)

        t = ""
        while(True):
            if(txt == []):
                break
            t += txt.pop(0)
        d3 = float(t)
        
        data += [[d1,d2,d3]]
        trainingPoints += 1
    print("Loaded " + str(trainingPoints) + " training examples!")


def getDataPoint():
    # Get random training point
    return data[math.floor(rand.random()*len(data))]


def sig(n):
    # Sigmoid function
    return (2 / (1 + (math.e ** -2*n))) - 1


def sigDeriv(n):
    # Derivative of sigmoid function
    return 4*(math.e ** -2*n)/((1 + math.e ** -2*n)**2)


def forwardProp(dataPoint,verbose):

    # Get output of NN given data point of the form [Input 1,Input 2]

    neur1[0] = dataPoint[0]
    neur1[1] = dataPoint[1]

    for i in range(0,len(neur2)):
        total = 0
        for j in range(0,len(weights1[i])):
            total += weights1[i][j] * neur1[j]
        total += biases1[i]
        
        neur2[i] = sig(total)

    for i in range(0,len(neur3)):
        total = 0
        for j in range(0,len(weights2[i])):
            total += weights2[i][j] * neur2[j]
        total += biases2[i]
        
        neur3[i] = sig(total)

    if(verbose):
        print("The AI thinks " + str(dataPoint[0]) + " and " + str(dataPoint[1]) + " should give " + str(neur3[0]))

    return neur3[0]


def backProp(dataPoint,verbose):

    # Get errors of final layer and backpropagate them to weights/biases of earlier layers
    # then adjust weight/biases to reduce error

    t1 = 0
    t2 = 0
    for i in range(0,len(weights1)):
        for j in range(0,len(weights1[i])):
            t1 += weights1[i][j]
    for i in range(0,len(weights2)):
        for j in range(0,len(weights2[i])):
            t2 += weights2[i][j]

    # Get erros of output nodes
    for i in range(0,len(neur3)):
        costs2[i] =  dataPoint[2] - neur3[i]
    if(verbose):
        print("Output layer costs: " + str(costs2))

    # Adjust weights/biases between layer 2 and 3
    for i in range(0,len(weights2)):
        for j in range(0,len(weights2[i])):
            weights2[i][j] -= (weights2[i][j]/t2) * costs2[i] * sigDeriv(neur3[i]) * learningRate
        biases2[i] -= (biases2[i]/t2) * costs2[i] * sigDeriv(neur3[i]) * learningRate

    # Get errors of layer 2 nodes
    for i in range(0,len(weights2)):
        total = 0
        for j in range(0,len(weights2[i])):
            total += costs2[i] * weights2[i][j]
        costs1[i] = total
    if(verbose):
        print("Hidden layer 1 costs: " + str(costs1))

    # Adjust weights/biases between layer 1 and 2
    for i in range(0,len(weights1)):
        for j in range(0,len(weights1[i])):
            weights1[i][j] -= (weights1[i][j]/t1) * costs1[i] * sigDeriv(neur2[i]) * learningRate
        biases1[i] -= (biases1[i]/t2) * costs1[i] * sigDeriv(neur2[i]) * learningRate


    if(verbose):
        print("")
        vars()
        print("")


def train(cycles,verbose=False):

    # Run forward and back propagation in succession [cycles] num of times
    print("Training AI...")
    for i in range(0,cycles):
        point = getDataPoint()
        forwardProp(point,verbose)
        backProp(point,verbose)
    print("Done!")


def test(inputPoint):
    # Print the output of forwardProp
    out = forwardProp(inputPoint,False)
    print(str(inputPoint[0])+","+str(inputPoint[1])+" -> "+str(out))


def vars():
    print("Weights1 = " + str(weights1))
    print("")

    print("Weights2 = " + str(weights2))
    print("")

    print("Biases1 = " + str(biases1))
    print("")

    print("Biases2 = " + str(biases2))

init()
