import torch
import rosbag
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import h5py
import pickle


def bag_to_tensor(bagfile, width, height, length):
    bag = rosbag.Bag(bagfile)
    tens = torch.zeros((length, height, width))

    for i, (topic, msg, t) in enumerate(bag.read_messages()):
        if i == length:
            break

        events = msg.events
        for ev in events:
            tens[i, ev.y, ev.x] = 1 if ev.polarity else -1

        if not i % 10:
            print(i//10, end=" ", flush=True)
    
    print()

    return tens


def mp4_to_tensor(vid, max_len=None):
    capture = cv2.VideoCapture(vid)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print(max_len)

    if max_len:
        length = min(max_len, video_length)
    else:
        length = video_length

    tens = np.empty((length, height, width), dtype=np.ubyte)
    frame_count = 0
    has_next_frame = True

    while (frame_count < length and has_next_frame):
        has_next_frame, rgb = capture.read()
        tens[frame_count] = np.sum(rgb, -1) // 3
        frame_count += 1
        print(frame_count, end=" ", flush=True)

    capture.release()

    return torch.from_numpy(tens)


def event_frames_per_video_frame(bagfile, mp4file):
    bag = rosbag.Bag(bagfile)
    timestamps = []

    for topic, msg, t in bag.read_messages():
        timestamps.append(msg.events[0].ts.nsecs)
        if len(timestamps) == 2:
            break

    event_fps = 1e9 / event_frame_dt(bagfile)
    mp4_fps = cv2.VideoCapture(mp4file).get(cv2.CAP_PROP_FPS)

    return event_fps / mp4_fps


# calculates the time delta between event timestamps, assuming it to be constant 
def event_frame_dt(bagfile):
    bag = rosbag.Bag(bagfile)
    t1 = 1e12

    for topic, msg, t in bag.read_messages():
        for ev in msg.events:
            t2 = ev.ts.nsecs
            if (dt := t2 - t1) > 0:
                return dt
            else:
                t1 = t2


def save_event_tensor(bagfile, shape, outname):
    event_tensor = bag_to_tensor(bagfile, *shape)
    with open('tensors/' + outname + '.evt', 'wb') as f:
        pickle.dump(event_tensor, f)


def save_mp4_tensor(mp4file, length, outname):
    mp4_tensor = mp4_to_tensor(mp4file, length)
    with open('tensors/' + outname + '.vdt', 'wb') as f:
        pickle.dump(mp4_tensor, f)


def load_tensor(file):
    with open('tensors/' + file, 'rb') as f:
        return pickle.load(f)


def main():
    bagfile = 'datasets/back6.bag'
    vid = 'datasets/back6.mp4'
    width = 640
    height = 480
    length = 105

    # save_event_tensor(bagfile, (width, height, length), 'short')

    save_mp4_tensor(vid, 3, 'short')
    vt = load_tensor('short.vdt')

    plt.imshow(vt[0])
    plt.show()


if __name__ == '__main__':
    main()
