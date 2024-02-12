import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_size[0]*2, out_channels=64, kernel_size=3, stride=1, padding=1)
 
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Define the GRU layer
        self.gru = nn.GRU(input_size=64*input_size[1]*input_size[2], hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, x, prev_enc_out):
        #add dimension to the encoder output 
        #prev_enc_out = prev_enc_out.unsqueeze(3)
        #prev_enc_out = prev_enc_out.expand(-1, -1, -1, 3)
        # Concatenate the input state with the previous enc_out
        x = torch.cat((x, prev_enc_out), dim=1)
        
        # Apply convolutional layers
        x = self.relu(self.conv1(x))
      
        x = self.flatten(x)
        
        # Reshape for GRU input
        x = x.view(x.size(0),1 , -1)
        
        # Apply GRU layer
        _, enc_out = self.gru(x)
        enc_out = enc_out.squeeze(0).unsqueeze(2).unsqueeze(3).expand(-1,-1,3,3)
        return enc_out