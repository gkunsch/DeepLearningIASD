#Try the softmax activation at the output with cross-entropy loss

import numpy as np

def naive_neural_network_classification_multi(steps, input_1,input_2,target_output1,target_output2):
    np.random.seed(seed=42)
    w1 = np.random.rand(1)
    w2 = np.random.rand(1)
    w3 = np.random.rand(1) 
    w4 = np.random.rand(1) #weight initialization
    
    def sigmoid(t): 
        return 1/(1+np.exp(-t))
    
    def softmax(x,y):
        return np.exp(x)/(np.exp(x)+np.exp(y)),np.exp(y)/(np.exp(x)+np.exp(y))
        
    def cross_entropy_loss(y1,o1,y2,o2): #for multi classification, attention pour avoir une valeur positive
        return -y1*np.log(o1)-(y2)*np.log(o2)-(1-y1)*np.log(1-o1)-(1-y2)*np.log(1-o2)
    
    for i in range (steps):
        n1 = input_1*w1+input_2*w2
        o1 = sigmoid(n1)
        n2 = o1*w3
        n3 = o1*w4
        o2, o3 = softmax(n2,n3)
        
        error = cross_entropy_loss(target_output1,o2, target_output2, o3)
        
        #backpropagation
        eta = 1
        do3 = o3 - target_output2
        do2 = o2 - target_output1

        grad_weight_4 = do3*o1
        grad_weight_3 = do2*o1
        grad_weight_2 = do2*w3*o1*(1-o1)*input_2
        grad_weight_1 = do2*w3*o1*(1-o1)*input_1
        
        w4 = w4 - eta*grad_weight_4
        w3 = w3 - eta*grad_weight_3
        w2 = w2 - eta*grad_weight_2
        w1 = w1 - eta*grad_weight_1
        
    return [error,w1,w2,w3,w4,o2,o3]

print(naive_neural_network_classification_multi(steps=1000,input_1=0.1,input_2=0.5,target_output1=1.0,target_output2=0.0))