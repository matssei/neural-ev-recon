import torch

class EventNet:
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = torch.nn.Conv