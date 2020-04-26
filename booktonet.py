"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import mnist_loader
import network

# Third-party libraries
import numpy as np

mems = 0

class Network():

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        global mems
        
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
            mems += 3*len(a)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        global mems
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        mems += 2*len(self.weights)
                        
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]
                       
        mems += 2*len(self.weights)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        global mems
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            mems += 3*len(w)
            
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime_vec(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        mems += 2* len(delta)
        
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            mems += 2*len(delta)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) 

#### Miscellaneous functions
def sigmoid(z):
#    """The sigmoid function."""
#    return max(z, 0)
#    if z < 50:
#        return np.log(1 + np.exp(z))
#    else:
#        return z
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
#    return 1*(z>0)
#    if z < 50:
#        return np.exp(z)/(np.exp(z) + 1)
#    else:
#        return 1
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 30, 10])

#print training_data
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# Neural net code here!

############################################################

# Conversion code here!

class Node:
    def __init__(self, nodeType, layer, ID, amInputNode=False, amOutputNode=False):
        self.nodeType = nodeType
        self.layer = layer
        self.ID = ID 
        self.amInputNode = amInputNode
        self.amOutputNode = amOutputNode

    def toSpiceNetlist(self, weightDictionary, inputArray):
        marker = str(self.layer) + "." + str(self.ID)
        
        outString = "* Circuitry for Node " + marker + "\n\n"
        
        if self.amInputNode:                    
            outString += "I" + marker + " 0 " + marker + "a " + str(float(inputArray[self.ID])) + "mA\n"
            outString += "V" + marker + " " + marker + "a " + marker + "b 0\n"
            outString += "R" + marker + " 0 " + marker + "b 1k\n\n"
                        
        else:       
            for otherNodeID in weightDictionary[self.layer][self.ID]:
                prevMarker = str(self.layer-1) + "." + str(otherNodeID)
                fullMarker = str(self.layer) + "." + str(otherNodeID) + "." + str(self.ID)
            
                outString += "F" + fullMarker + " 0 " + marker + "a V" + prevMarker + " " + \
                     str(weightDictionary[self.layer][self.ID][otherNodeID]) + "\n"
            
            if self.nodeType == "ReluLayer":         
                outString += "R" + marker + " 0 " + marker + "a 1000k\n"
                
            outString += "V" + marker + " " + marker + "a " + marker + "b 0\n"
            
            if self.nodeType == "ReluLayer":
                outString += "D" + marker + " " + marker + "b " + marker + "c 1mA_diode\n"
            elif self.nodeType == "LinearLayer":
                outString += "V" + marker + "b " + marker + "b " + marker + "c 0\n"
            
            if self.amOutputNode:
                outString += "Rout." + str(self.ID) + " " + marker + "c 0 1k\n\n"
            else:
                outString += "V" + marker + "c " + marker + "c 0 0\n\n"
                
        return outString
		
def neuralNetToSpiceNetlist(weightDictionary, nodeDictionary, inputArray):
    outString = ""
    
    for i in nodeDictionary:
        for j in nodeDictionary[i]:    
            outString += nodeDictionary[i][j].toSpiceNetlist(weightDictionary, inputArray)
    
    outString += ".model 1mA_diode D (Is=100pA n=1.679)\n"
    
    output = open("/Users/adamyedidia/Documents/MacSpice/neuralnet.cir", "w")
    
    output.write(outString)

def makeNeuralNet(net):
	numLayers = len(net.weights) + 1
	
	nodeDictionary = {}
	weightDictionary = {}
	
	for i in range(numLayers):
		nodeDictionary[i] = {}
		weightDictionary[i+1] = {}
		
	for i in range(numLayers-1):		
		for k in range(len(net.weights[i])):
			weightDictionary[i+1][k] = {}
	
	for i in range(numLayers):
		if not i == numLayers - 1:
			
			if i == 0:
				for j in range(len(net.weights[i][0])):
					nodeDictionary[i][j] = Node("ReluLayer", i, j, True, False)
					for k in range(len(net.weights[i])):
						weightDictionary[i+1][k][j] = net.weights[i][k][j]
					
			else:
				for j in range(len(net.weights[i][0])):
					nodeDictionary[i][j] = Node("ReluLayer", i, j, False, False)
					for k in range(len(net.weights[i])):
						weightDictionary[i+1][k][j] = net.weights[i][k][j]
						
		else:
			for j in range(len(net.weights[-1])):
				nodeDictionary[i][j] = Node("LinearLayer", i, j, False, True)
						
	return weightDictionary, nodeDictionary				
			
weightDictionary, nodeDictionary = makeNeuralNet(net)
inputArray = test_data[0][0]
neuralNetToSpiceNetlist(weightDictionary, nodeDictionary, inputArray)
#print test_data[0][0]

print net.feedforward(test_data[0][0])
print "Memory accesses:", mems