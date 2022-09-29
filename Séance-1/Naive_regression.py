
#Network with two inputs, one output, one hidden layer of one neuron.
#Sigmoid activation function. No bias.
#Inputs are 0.1 and 0.5, desired output is 0.2. Write code for the forward pass.
#Compute the mean squared error. Backpropagate the error.
#Train the network.

import numpy as np 

def naive_neural_network_regression(n, input_1,input_2,target_output):
    
    np.random.seed(seed=42)
    w1 = np.random.rand(1)
    w2 = np.random.rand(1)
    w3 = np.random.rand(1) #weight initialization 

    def sigmoid(t): 
        return 1/(1+np.exp(-t))

    def mean_squared_error(x,y):
        return 0.5*(x-y)**2
    
    for i in range (n):
        n1 = input_1*w1+input_2*w2
        o1 = sigmoid(n1)
        n2 = o1*w3
        o2 = sigmoid(n2) 

        error = mean_squared_error(o2,target_output)

        #backpropagation -> chain rule
        eta = 1
        grad_weight_3 = -(target_output-o2)*o2*(1-o2)*o1
        grad_weight_2 = -(target_output-o2)*o2*(1-o2)*w3*o1*(1-o1)*input_1
        grad_weight_1 = -(target_output-o2)*o2*(1-o2)*w3*o1*(1-o1)*input_2

        w3 = w3 - eta*grad_weight_3
        w2 = w2 - eta*grad_weight_2
        w1 = w1 - eta*grad_weight_1
        
    return [error,w1,w2,w3,o2]

print(naive_neural_network_regression(1000,0.1,0.5,0.2))