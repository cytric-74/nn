import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# flattening the 2d matrice
def flatten(maze):
    return [cell for row in maze for cell in row]

def init_network(input_size, hidden_sizes, output_size):
    layers = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    for i in range(len(layer_sizes)-1):
        # used the xavier initialization and scaling and weight balancing
        scale = math.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
        weights = [[random.uniform(-scale, scale) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])]
        biases = [random.uniform(-scale, scale) for _ in range(layer_sizes[i+1])]
        layers.append({'weights': weights, 'biases': biases})
    return layers

def forward_pass(network, input_vector):
    activations = input_vector
    all_layers = [activations]

    for layer in network:
        next_activations = []
        for weights, bias in zip(layer['weights'], layer['biases']):
            z = sum(w * a for w, a in zip(weights, activations)) + bias
            next_activations.append(sigmoid(z))
        activations = next_activations
        all_layers.append(activations)
    
    return all_layers

def backward_pass(network, all_layers, expected, learning_rate):
    deltas = []
    layers_count = len(network)
    
    # error handling at output layer
    output = all_layers[-1]
    error = [(expected[i] - output[i]) * sigmoid_derivative(output[i]) for i in range(len(expected))]
    deltas.append(error)

    # for handling hidden layer error
    for l in range(layers_count-1, 0, -1):
        delta = []
        current_activations = all_layers[l]
        for i in range(len(network[l-1]['weights'])):
            error = sum(network[l]['weights'][j][i] * deltas[0][j] for j in range(len(network[l]['weights'])))
            delta.append(error * sigmoid_derivative(current_activations[i]))
        deltas.insert(0, delta)
    
    # updating
    for l in range(len(network)):
        for i in range(len(network[l]['weights'])):
            for j in range(len(network[l]['weights'][i])):
                network[l]['weights'][i][j] += learning_rate * deltas[l][i] * all_layers[l][j]
            network[l]['biases'][i] += learning_rate * deltas[l][i]

