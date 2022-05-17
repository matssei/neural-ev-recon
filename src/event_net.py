import torch
import pickle

from tensor_generation import *

class EventNet(torch.nn.Module):
    def __init__(self):
        super(EventNet, self).__init__()
        self.event_frames_per_video_frame = 35
        self.video_shape = (1080, 1920)
        self.conv_kernel_size = 9
        self.unfold_kernel_size = 40
        self.layers = 1

        self.video_size = self.video_shape[0] * self.video_shape[1]
        self.batch_size = self.unfold_kernel_size**2
        self.hidden_size = self.video_size // self.batch_size

        self.hidden_state = torch.rand(self.layers, self.batch_size, self.hidden_size)

        self.conv = torch.nn.Conv2d(self.event_frames_per_video_frame, 1, self.conv_kernel_size, padding='same')
        self.up = torch.nn.Upsample(self.video_shape)
        self.unfold = torch.nn.Unfold(kernel_size=self.unfold_kernel_size, stride=self.unfold_kernel_size)
        self.rnn = torch.nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        x = self.unfold(x)
        x, self.hidden_state = self.rnn(x, self.hidden_state)
        self.hidden_state.detach_()
        x = x.reshape(-1, *self.video_shape)
        
        return x

    def loss(self, x, y):
        return torch.sqrt(torch.mean(torch.square(x.squeeze() - y)))


def get_training_tensors(event_bag, event_shape, mp4):
    events = bag_to_tensor(event_bag, event_topic, *event_shape)
    frames = mp4_to_tensor(mp4)
    
    return events, frames


def save_net(net):
    name = input('Net name: ')
    with open('nets/' + name, 'wb') as f:
        pickle.dump(net, f)


def main():
    width = 640
    height = 480
    ev_per_fr = 35

    et = load_tensor('28frames.evt')
    vt = load_tensor('28frames.vdt')

    et = et.reshape((-1, ev_per_fr, width, height))

    net = EventNet()
    loss = net.loss
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

    stop = 5

    for i in range(1_000_000_000):
        out = net(et)
        l = loss(out, vt)
        print(f'Epoch {i}   \t\tLoss = {l.data.item()}')
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if i == stop:
            plt.imshow(out.detach().numpy()[0])
            plt.show()

            print('[s]ave net and exit, [e]xit, or [c]ontinue')

            while (choice := input()) not in ['s', 'e', 'c']:
                print('Invalid input')

            if choice == 's':
                save_net(net)
                break

            if choice == 'e':
                break

            if choice == 'c':
                stop = int(input('Epoch for next stop: '))


if __name__ == '__main__':
    main()