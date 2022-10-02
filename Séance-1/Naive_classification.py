# Train the same network to have 1.0 as output with the binary cross-entropy loss.
import numpy as np 

def naive_neural_network_classification(n, input_1,input_2,target_output):
    np.random.seed(seed=42)
    w1 = np.random.rand(1)
    w2 = np.random.rand(1)
    w3 = np.random.rand(1) #weight initialization 

    def sigmoid(t): 
        return 1/(1+np.exp(-t))

    def binary_cross_entropy_loss(y,o): #for binary classification
        return -y*np.log(o)-(1-y)*np.log(1-o)
    
    for i in range (n):
        n1 = input_1*w1+input_2*w2
        o1 = sigmoid(n1)
        n2 = o1*w3
        o2 = sigmoid(n2) 

        error = binary_cross_entropy_loss(target_output,o2)
        #backpropagation
        eta = 1
        grad_weight_3 = (-target_output/o2+(1-target_output)/(1-o2))*o2*(1-o2)*o1
        grad_weight_2 = (-target_output/o2+(1-target_output)/(1-o2))*o2*(1-o2)*w3*o1*(1-o1)*input_2
        grad_weight_1 = (-target_output/o2+(1-target_output)/(1-o2))*o2*(1-o2)*w3*o1*(1-o1)*input_1

        w3 = w3 - eta*grad_weight_3
        w2 = w2 - eta*grad_weight_2
        w1 = w1 - eta*grad_weight_1
        
    return [error,w1,w2,w3,o2]

print(naive_neural_network_classification(1000,0.1,0.5,1.0))