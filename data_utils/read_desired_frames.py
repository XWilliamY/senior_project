import numpy as np

def read_desired_frames(filename, frame_rate, target_frame_rate):
    tuples = []
    print(filename)
    with open(filename, 'r') as f:
        temp = f.read().splitlines()
        for line in temp:
            # change 1 and 2 here
            desired = [int(i) for i in line.split(',')]
            desired[1] = int(np.ceil(desired[1] / frame_rate) * target_frame_rate)
            desired[2] = int(np.floor(desired[2] / frame_rate) * target_frame_rate) - 1
            adjusted_desired = tuple(desired)
            tuples.append(adjusted_desired)
    return tuples
