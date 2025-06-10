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

    for i in range(2):
        summation = sum(inputs[j] * net["input_hidden_weights"][j][i] for j in range(2)) + net["hidden_bias"][i]
        hidden_in.append(summation)
        hidden_out.append(sigmoid(summation))

    final_input = sum(hidden_out[i] * net["hidden_output_weights"][i] for i in range(2)) + net["output_bias"]
    final_output = sigmoid(final_input)

    return hidden_out, final_output

def train(network, data, labels, epochs, learning_rate):
    for epoch in range(epochs):
        total_error = 0
        for idx in range(len(data)):
            inputs = data[idx]
            expected = labels[idx]

            hidden_output, final_output = forward_pass(inputs, network) # forward passing
            error = expected - final_output
            total_error += error ** 2

            # backward passing
            d_output = error * sigmoid_derivative(final_output)

            d_hidden = [d_output * network["hidden_output_weights"][i] * sigmoid_derivative(hidden_output[i]) for i in range(2)]

            for i in range(2):  # hidden to output
                network["hidden_output_weights"][i] += learning_rate * d_output * hidden_output[i]

            network["output_bias"] += learning_rate * d_output

            for i in range(2):  # input to hidden
                for j in range(2):
                    network["input_hidden_weights"][j][i] += learning_rate * d_hidden[i] * inputs[j]

                network["hidden_bias"][i] += learning_rate * d_hidden[i]

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, MSE: {total_error/len(data)}")