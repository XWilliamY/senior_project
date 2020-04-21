def read_desired_frames(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line.split(',')[0])

read_desired_frames('trial.txt')

'''
turn the text from a bunch of strings into

[(id, frame_begin, frame_end), ...
 (id, frame_begin, frame_end)]

id is array, frame_begin and frame_end are integers

'''
