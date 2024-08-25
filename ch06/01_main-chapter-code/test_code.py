import torch
from torch.nn.functional import cross_entropy

input = torch.randn(3, 5, requires_grad=True)
# convert input to numpy arry

target = torch.randint(5, (3,), dtype=torch.int64)
loss = cross_entropy(input, target)
loss.backward()
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
loss = cross_entropy(input, target)
loss.backward()

print("Success")