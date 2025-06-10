# Simple Neural Network from Scratch (No Libraries)

This is a very basic neural network written entirely in Python. It's just Python and math to help me understand how a neural network actually works under the hood.

## What it does

This small neural net:
- Has 2 input neurons
- 1 hidden layer with 2 neurons
- 1 output neuron
- Uses sigmoid activation
- Trains using backpropagation
- Solves the XOR logic gate problem

It runs a forward pass to predict, calculates the error, and then adjusts the weights using gradient descent during backpropagation.

## Training Data

We're using XOR as training data:
``` Input: [0, 0] -> Output: 0
Input: [0, 1] -> Output: 1
Input: [1, 0] -> Output: 1
Input: [1, 1] -> Output: 0
```


