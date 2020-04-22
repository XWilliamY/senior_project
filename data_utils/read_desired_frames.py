def read_desired_frames(filename):
    tuples = []
    with open(filename, 'r') as f:
        temp = f.read().splitlines()
        print(temp)
        for line in temp:
            desired = tuple(int(i) for i in line.split(','))
            tuples.append(desired)
    return tuples
