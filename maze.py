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



