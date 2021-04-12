import torch as T
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,lr=1e-3, input_dim=2):
        super().__init__()
        
        # Initialize Input dimentions
        self.input_dim = 2
        self.input_fc1_dim = 8
        self.fc1_output_dim = 8
        self.output_dim = 2
        
        # Define the NN layers
        self.input_layer = nn.Linear(self.input_dim, self.input_fc1_dim)
        self.fc1 = nn.LeakyReLU(0.1)
        self.output_layer = nn.Linear(self.fc1_output_dim, self.output_dim)
        
        # Initialize layers weights
        self.input_layer.weight.data.normal_(0,1)
        #self.fc1.weight.data.normal_(0,0.2)
        self.output_layer.weight.data.normal_(0,1)
        
        # Initialize layers biases
        nn.init.constant_(self.input_layer.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
        
        self.lr = lr
        # Define Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.lr)
        
        # Set Device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input, num_particle=5):
        X = self.input_layer(input)
        #X = F.relu(X)
        X = self.fc1(X)
        #X = F.relu(X)
        output = self.output_layer(X)
        #output = F.relu(output)
        return output
    
class Discriminator(nn.Module):
    def __init__(self,lr=1e-4, input_dim=2):
        super().__init__()
        
        # Initialize Input dimentions
        self.input_dim = 2
        self.input_fc1_dim = 64
        self.fc1_output_dim = 64
        self.output_dim = 1
        
        # Define the NN layers
        self.input_layer = nn.Linear(self.input_dim, self.input_fc1_dim)
        self.fc1 = nn.LeakyReLU(0.1)
        self.output_layer = nn.Linear(self.fc1_output_dim, self.output_dim)
        
        # Initialize layers weights
        self.input_layer.weight.data.normal_(0,1)
        #self.fc1.weight.data.normal_(0,0.2)
        self.output_layer.weight.data.normal_(0,1)
        
        # Initialize layers biases
        nn.init.constant_(self.input_layer.bias, 0.0)
        #nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
        
        self.lr = lr
        # Define Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.lr)
        
        # Set Device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input):
        X = self.input_layer(input)
        #X = F.LeakyReLU(X)
        X = self.fc1(X)
        #X = F.relu(X)
        output = self.output_layer(X)
        return output