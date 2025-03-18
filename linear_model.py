import numpy 
import matplotlib.pyplot as plt
import torch
from torch import nn

class Liner_model(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.weight = nn.Parameter()
    
    # def __init__(self,x_user_input,y_user_input):
    #     self.input_data = x_user_input
    #     self.label_data = y_user_input
    #     super
        
        
            
    def data_loader(self):
        start = 0
        end = 1
        label_weight = 0.7
        label_bias = 0.3
        
        
        x = torch.arange(start=start,end=end,step=self.input_data)
        y = label_weight*self.input_data + label_bias
        
        split  = int(len(x)*0.80)
        
        x_train_split = x[:split]
        y_train_split = y[:split]
        
        x_test_split = x[split:]
        y_test_split = y[split:]
        

        
        
        