import random

def allListsOfSizeX(x):
    if x == 0:
        return [[]]
        
    else:
        oneLess = allListsOfSizeX(x-1)
        return [i + [0] for i in oneLess] + [i + [1] for i in oneLess]  

def pront(x):
    print x

def convertToOneMinusOne(x):
    if x == 0:
        return -1.0
        
    return 1.0
    
def majority(inputs, params):
    assert len(inputs) == len(params)
    
    counter = 0
    
    for inp, param in zip(inputs, params):
        counter += convertToOneMinusOne(inp) * param
        
    return 1*(counter>0)

def dot(inputs, params):
    assert len(inputs) == len(params)
    
    counter = 0
    
    for inp, param in zip(inputs, params):
        counter += convertToOneMinusOne(inp) * param
        
    return counter

class Majoritarian:
    def __init__(self, functionList):
        self.functionList = functionList
        self.numParams = len(functionList)            
                
    def evaluate(self, inputs):
        inputsToMajority = [f(inputs) for f in self.functionList]
        
        return [majority(inputsToMajority, self.params)]
        
    def evaluateReal(self, inputs):
        inputsToMajority = [f(inputs) for f in self.functionList]
        
        return [dot(inputsToMajority, self.params)]
                
    def evaluateRandom(self, inputs, numSamples=999):        
        votesForOne = 0
        votesForZero = 0
        
        for _ in range(numSamples):
        
            dictatorIndex = random.randint(0, self.numParams-1)
            
            f = self.functionList[dictatorIndex]
            
            votesForOne += (convertToOneMinusOne(f(inputs)) == self.params[dictatorIndex])
            votesForZero += (convertToOneMinusOne(f(inputs)) != self.params[dictatorIndex])
        
        return [votesForOne > votesForZero]
        
    def train(self, trainingSet, hardCutoffs=False):
        # Note: only works on 1-output functions
        
        self.params = [0]*len(self.functionList)
        
        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            output = dataPoint[1][0]
            
            inputsToMajority = [f(inputs) for f in self.functionList]
            
            for i, majInput in enumerate(inputsToMajority):
 #               if majInput == output:
#                else:
#                    self.params[i] -= 1.0

                self.params[i] += convertToOneMinusOne(majInput)*output/len(trainingSet)

#                if majInput == output:
#                    self.params[i] += 1.0
#                else:
#                    self.params[i] -= 1.0
                    
        if hardCutoffs:            
            self.params = [1.0*(x>0) + -1.0*(x<0) for x in self.params]
#            self.params = [1.0*(convertToOneMinusOne(x>=0)) for x in self.params]
            
        print self.params
              
    def generateTruthTable(self, numInputs, realEvaluation=False):
        trainingSet = []
        
        for inp in allListsOfSizeX(numInputs):
            if realEvaluation:
                trainingSet.append([inp, self.evaluateReal(inp)])
            else:
                trainingSet.append([inp, self.evaluate(inp)])
            
        return trainingSet
        
    def test(self, testSet, realEvaluation=False, randomEvaluation=False, randomOutcomes=False, verbose=False):
        correctnessCounter = 0.0
        randomCounter = 0.0
        alwaysZeroCounter = 0.0
        overallCounter = 0.0
        
        for dataPoint in testSet:
            inputs = dataPoint[0]
            correctOutputs = dataPoint[1]
            
            if realEvaluation:
                myOutputs = self.evaluateReal(inputs)                
            else:
                if randomEvaluation:
                    myOutputs = self.evaluateRandom(inputs)                
                else:            
                    myOutputs = self.evaluate(inputs)
                                    
            if verbose:
                pront("Correct: " + str(correctOutputs))
                pront("Observed: " + str(myOutputs))
                pront("")                
                        
            for i in range(len(correctOutputs)):              
                if myOutputs[i] == correctOutputs[i]:
                    correctnessCounter += 1.0
                
                if random.random() < 0.5:
                    randomCounter += 1.0
				
                if not correctOutputs[i]:
                    alwaysZeroCounter += 1.0
                
                overallCounter += 1.0
        
        pront("Got " + str(correctnessCounter) + " out of " + str(overallCounter) + " correct.")
        pront("")
        
        if randomOutcomes:        
            pront("Compare to the random outcome: ")
            pront("Got " + str(randomCounter) + " out of " + str(overallCounter) + " correct.")	
            pront("")
            pront("Compare to the outcome you'd have gotten if you always picked zero: ")
            pront("Got " + str(alwaysZeroCounter) + " out of " + str(overallCounter) + " correct.")
				
        return correctnessCounter / overallCounter      