import numpy 
import matplotlib.pyplot as plt
import torch
from torch import nn

class load_visulization():
    def __init__(self,input_step):
        self.input_step = input_step
            
    def data_loader(self):
        start = 0
        end = 1
        label_weight = 0.7
        label_bias = 0.3
        
        
        self.x = torch.arange(start=start,end=end,step=self.input_step).unsqueeze(dim=1)
        self.y = label_weight*self.x + label_bias
        
        split  = int(len(self.x)*0.80)
        
        self.x_train_split = self.x[:split]
        self.y_train_split = self.y[:split]
        
        self.x_test_split = self.x[split:]
        self.y_test_split = self.y[split:]
        
    def visulization(self,prediction):
        
       plt.figure(figsize=(10,7),dpi=150)
       
       plt.scatter(x=self.x_train_split,
                   y=self.y_train_split,
                   c="blacl",s=3,label="train-set")
       
       plt.scatter(x=self.x_test_split,
                   y=self.y_test_split,
                   c="blue",s=4,label="test-set")
       
       if prediction is not None:
       
            plt.scatter(x=self.x_test_split,
                        y=prediction,
                        c="green",s=2,label="model-output")
            
            
    plt.xlabel("data")
    plt.ylabel("label")
    plt.title("model_data")
    plt.legend()
    plt.show()
