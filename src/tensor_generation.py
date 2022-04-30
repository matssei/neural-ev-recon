import torch
import rosbag
import matplotlib.pyplot as plt
import time


def load_event_tensor_from_bag(bagfile, topic, width, height, length):
    bag = rosbag.Bag(bagfile)
    tens = torch.zeros((width, height, length))

    for i, (topic, msg, t) in enumerate(bag.read_messages()):
        if i == length:
            break

        events = msg.events
        for ev in events:
            tens[ev.x, ev.y, i] = 1 if ev.polarity else -1

        if not i % 10:
            print(i//10, end=" ", flush=True)
    
    print()

    return tens


def display_tensor(tens):
    for frame in tens:
        plt.imshow(frame.numpy())
        plt.show()
        time.sleep(0.2)


def main():
    bagfile = 'datasets/back6.bag'
    topic = '/dvs/cam1/events'
    width = 640
    height = 480
    length = 100

    t = load_event_tensor_from_bag(bagfile, topic, width, height, length)


if __name__ == '__main__':
    main()
