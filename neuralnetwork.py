import math
import random

def sigmoid(x):
    return 1/(1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (x-1)

def init_networking():
    network = {
        "input_hidden_weights": [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)],
        "hidden_output_weights": [random.uniform(-1, 1) for _ in range(2)],
        "hidden_bias": [random.uniform(-1, 1) for _ in range(2)],
        "output_bias": random.uniform(-1, 1)
    }
    return network

def forward_pass(inputs, net):
    hidden_in = []
    hidden_out = []

    for i in range(2)
        summation = sum(inputs[j] * net["input_hidden_weights"][j][i] for j in range(2)) + net["hidden_bias"][i]
        hidden_in.append(summation)
        hidden_out.append(sigmoid(summation))

    final_input = sum(hidden_out[i] * net["hidden_output_weights"][i] for i in range(2)) + net["output_bias"]
    final_output = sigmoid(final_input)

    return hidden_out, final_output