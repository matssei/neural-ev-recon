import torch

from tensor_generation import *

class EventNet(torch.nn.Module):
    def __init__(self):
        super(EventNet, self).__init__()
        self.conv = torch.nn.Conv2d(35, 20, 10)

    def forward(self, x):
        x = self.conv(x)


def get_training_tensors(event_bag, event_shape, mp4):
    events = bag_to_tensor(event_bag, event_topic, *event_shape)
    frames = mp4_to_tensor(mp4)
    
    return events, frames


def main():
    bagfile = 'datasets/back6.bag'
    vid = 'datasets/back6.mp4'
    width = 640
    height = 480
    length = 100

    #get_training_tensors(bagfile, topic, (width, height, length), vid)

    n_event_frames = event_frames_per_video_frame(bagfile, vid)
    print(n_event_frames)

    net = EventNet()


if __name__ == '__main__':
    main()