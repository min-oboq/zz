# import torch.nn as nn
# import torch

# madel = nn.Linear(1,1)

# class SingleLayer(nn.Modile):
#     def __init__(self, inputs):
#         super(SingleLayer, self).__init__()
#         self.layer = nn.Linear(inputs, 1)
#         self.activation = nn.Sigmoid()
        
#     def forward(self, X):
#         X = self.layer(X)
#         X = self.activation(X)
#         return X
    
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#         nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#         nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(2))
#         self.layer3 = nn.Sequential(
#             nn.Linear(in_features=30*5*5, out_features=10, bias=True),
#             nn.ReLU(inplace=True))
        
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = x.view(x.shape[0], -1)
#         x = self.layer3(x)
# def main():
#     model = MPL()
# print
    
# if __name__=='__main__':
# main
# print(model.weight)
# print(model.bias)