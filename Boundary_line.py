import numpy as np
def step(weighted_sum): # step activation function
    # The step activation is applied to the perceptron output that
    # returns 0 if the weighted sum is less than 0 and 1 otherwise
    return (weighted_sum > 0) * 1
 
def forward_propagation(input_data, weights, bias):
    #Computes the forward propagation operation of a perceptron 
    # and returns the output after applying the step activation function
  
   # takes the dot product of input and the weights and adds the bias
    return step(np.dot(input_data, weights) + bias) 

# Initialize parameters
X = np.array([2, 3]) # declaring two data points
Y = np.array([0]) # label
weights = np.array([2.0, 3.0]) # weights of perceptron
bias = 0.1 # bias value
Y_predicted = forward_propagation(X, weights.T, bias) # predicted label
print("Predicted label:", Y_predicted)