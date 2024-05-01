import math
from matplotlib import pyplot as plt

BIAS = 1
LEARNING_RATE = 0.1

inputs = [1, 0.5, 1, 0.75]
layers = []
outputs = []
original_t_outputs = []
target_outputs = [1, 0]
weights = []
squared_errors = []
squared_error_points = []

def Sigmoid(value):
    return 1/(1+math.exp(-value))

def Softmax(value):
    denom = 0

    for neuron in layers[-1]:
        denom += math.exp(neuron.getTotal())

    return math.exp(value)/denom

class Perceptron:
    def __init__(self, layer, weights=[]):
        self.total = 0
        self.inputs = []
        self.weights = weights
        self.layer = layer

        if len(layers) < layer:
            layers.append([])

        layers[layer - 1].append(self)

    # Update weights from delta weights
    def updateWeights(self, weight_update):
        new_weights = []

        for i in range(len(self.weights)):
            new_weights.append(self.weights[i] + (weight_update[i]))
        
        self.weights = new_weights.copy()

    def getWeight(self, index):
        return self.weights[index + 1]

    def getError(self):
        global target_outputs
        
        # Get error if output layer
        if self.layer == len(layers):
            error = target_outputs.pop(0) - self.getTotal()

            if len(target_outputs) == 0:
                target_outputs = original_t_outputs.copy()

            return error

            # Get error if hidden layer
        else:
            output = self.getActOut()
            self_index = layers[self.layer - 1].index(self)
            next_error_sum = 0
            
            # Get sum of weighted errors
            for perceptron in layers[self.layer]:
                next_error_sum += perceptron.getError()*perceptron.getWeight(self_index)

            # Return error
            return output * (1 - output) * next_error_sum

    def getInputs(self):
        return self.inputs

    def updateInputs(self):
        self.inputs.clear()
        self.inputs.append(BIAS)

        if self.layer == 1:
            self.inputs = inputs.copy()
        else:
            for perceptron in layers[self.layer - 2]:
                self.inputs.append(perceptron.getActOut())

    # Returns sum of inputs*weights
    def getTotal(self):
        self.total = 0

        for i in range(len(self.inputs)):
            self.total += self.inputs[i]*self.weights[i]
        
        return self.total

    def getActOut(self):
        return Sigmoid(self.getTotal())

    def getSoftOut(self):
        return Softmax(self.getTotal())

def getOutput():
    outputs.clear()

    for output in layers[-1]:
        outputs.append(output.getSoftOut())

    outputs_sorted = outputs.copy()
    outputs_sorted.sort()

    return outputs.index(outputs_sorted[-1]) + 1

def forward():
    for layer in layers:
        for perceptron in layer:
            perceptron.updateInputs()

    squared_errors.append(0.5 *((target_outputs[0] - layers[-1][0].getTotal())**2 + (target_outputs[1] - layers[-1][1].getTotal())**2))

def train():
    global LEARNING_RATE
    global inputs
    global original_t_outputs
    global target_outputs
    global squared_error_points
    global weights
    
    global squared_errors
    squared_errors.clear()

    # Read training data
    train_file = open("data-CMP2020M-item1-train.txt")
    train_data = []

    for line in train_file.readlines():
        line = line.replace('\n', '')
        data = line.split('\t')
        inputs = [float(x) for x in data[0].split().copy()]
        target_outputs = [float(x) for x in data[1].split().copy()]

        train_data.append([inputs, target_outputs])

    train_file.close()

    # Record current weights for table
    epoch_weights = []

    for layer in layers:
        for perceptron in layer:
            epoch_weights.append(perceptron.weights)

    # Train for each data sample
    for data in train_data:

        # Set inputs and target outputs of sample
        inputs = data[0].copy()
        inputs.insert(0, BIAS)
        target_outputs = data[1].copy()
        original_t_outputs = target_outputs.copy()

        # Do forward step with sample
        forward()

        weight_updates = []

        # Back propagation with sample
        for layer in list(reversed(layers)):
            for perceptron in layer:
                error = perceptron.getError()
                weight_update = []

                # Get delta weights
                for input in perceptron.getInputs():
                    weight_update.append(LEARNING_RATE * error * input)

                weight_updates.append(weight_update)

        # Update weights
        for layer in list(reversed(layers)):
            for perceptron in layer:
                perceptron.updateWeights(weight_updates.pop(0))

    squared_error_points.append(sum(squared_errors))
    weights.append(epoch_weights)

# Initiate network
INITIAL_PERCEPTRONS = [Perceptron(1, [0.9, 0.74, 0.8, 0.35]),
                Perceptron(1, [0.45, 0.13, 0.4, 0.97]),
                Perceptron(1, [0.36, 0.68, 0.1, 0.96]),
                Perceptron(2, [0.98, 0.35, 0.5, 0.9]),
                Perceptron(2, [0.92, 0.8, 0.13, 0.8])]

# Train for 200 epochs
for i in range(75):
    train()

# Print weight table
print("|Step\t", end="|")

for i in range(11):
    print("{}\t".format(i), end="|")
print("")
print("|-------"*12 + "|")

for i in range(len(weights[0][0]) + 1):
    for j in range(len(weights[0][0])):
        if i + len(inputs) < 7:
            print("|w{}{}\t".format(j, i + len(inputs)), end="|")
        else:
            if j > 0:
                print("|w{}{}\t".format(j + 3, i + len(inputs)), end="|")
            else:
                print("|w{}{}\t".format(j, i + len(inputs)), end="|")
        for epoch in range(11):
            print("{:0.3f}\t".format(weights[epoch][i][j]), end="|")

        print("")

# Print probability distribution of test data
inputs = [1, 0.3, 0.7, 0.9]
forward()
print("\nInput vector: [0.3, 0.7, 0.9]")
print("Output: {}".format(getOutput()))
print("Probabillity distrubition: {}".format(outputs))

# Plot and show graph
plt.plot(squared_error_points)
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Squared Error")
plt.show()