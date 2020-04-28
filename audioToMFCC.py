import librosa # for working with audio in python
import numpy as np
import argparse
from data_utils.check_dirs import check_input_dir, check_output_dir
from data_utils.read_desired_frames import read_desired_frames

def convert(audio_file, output_dir, targets, video_id):
    """
    time_in_seconds * sample_rate / hop_length = frame in mfcc
    """

    x, sample_rate = librosa.load(audio_file, sr=None)

    total_samples = x.shape[0]
    samples_per_second = sample_rate
    hop_length = 490
    mfccs = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40, hop_length=hop_length)

    if targets == None:
        save_to = output_dir + audio_file.split('/')[-1][:-4] + "_all_mfccs.npy"
        np.save(save_to, mfccs)
        return save_to

    # otherwise process targets
    line = 0
    for target_person_id, frame_begin, frame_end in targets:
        print("in audioToMFCC")
        print(frame_begin, frame_end)
        collect = []
        
        # frame_begin and frame_end adjusted to 30 fps already
        if frame_begin < 0:
            frame_begin = 0
        if frame_end > total_samples / sample_rate * 30: # last frame
            frame_end = total_samples / sample_rate * 30 

        # because of 0 indexing, need to increase frame_end by one
        frame_end += 1

        # convert from video frames into seconds into mfcc frames 
        initial = round(frame_begin / 30 * sample_rate / hop_length) # assume 512 as hop length
        end = round(frame_end/ 30 * sample_rate / hop_length)        # convert to seconds
        for i in range(initial, end)[::3]:
            j = i + 3
            if i > mfccs.shape[-1]:
                break
            sub_mfccs = mfccs[:, i : j]
            if (sub_mfccs.shape[-1] != 3):
                padded_sub_mfccs = np.zeros([40, 3])
                shape = sub_mfccs.shape
                padded_sub_mfccs[:shape[0], :shape[1]] = sub_mfccs
                sub_mfccs = padded_sub_mfccs

            collect.append(sub_mfccs)

        np_sub_mfccs = np.array(collect)
        np_sub_mfccs = np.transpose(np_sub_mfccs, (1, 0, 2))
        np_sub_mfccs = np_sub_mfccs.reshape(40, -1) # so librosa can read this
        save_to = f"{output_dir}mfcc_{video_id}_line_{line}.npy"
        np.save(save_to, np_sub_mfccs)
        print(np_sub_mfccs.shape)
        print(save_to)
        # reset
        line += 1
        collect = []

def main(args):
    # need input_dir
    if args.input_dir:
        input_dir = check_input_dir(args.input_dir)
    else:
        input_dir = check_output_dir(args.input_dir)

    # get frame rate of videos
    input_path = input_dir.split('/') # get frame rate
    video_id = input_path[-2]
    fps_txt = '/'.join(input_path) + video_id + "_fps.txt"
    target_frame_rate = 30
    frame_rate = None
    with open(fps_txt, 'r') as f:
        line = f.readline()
        frame_rate = int(float(line.split()[0]))

    # get targets
    targets = '/'.join(input_path) + video_id + "_targets.txt"
    desired_person_at_frame = read_desired_frames(targets, frame_rate, target_frame_rate)

    # get audio filename
    if args.input_audio_filename:
        audio_filename = input_dir + args.input_audio_filename
    else:
        audio_filename = input_dir + video_id + '.mp3'
    
    # lastly, direct it to data folder
    input_dir = check_output_dir(input_dir + 'data/')
    # output_dir can be optional, if not specified, default will be same as input_dir
    if not args.output_dir:
        output_dir = input_dir
    else:
        output_dir = check_output_dir(args.output_dir)

    # let convert handle the file saving
    convert(audio_filename, output_dir, desired_person_at_frame, video_id)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        help="Path to directory containing relevant npy data",                        
                        default=None,
                        type=str)

    parser.add_argument("--input_audio_filename",
                        help="Name of specific file to preprocess",
                        default=None,
                        type=str)
    
    parser.add_argument("--output_dir",
                        help="Path to output directory, default will be same as input_dir",
                        default=None,
                        type=str)

    parser.add_argument("--targets",
                        help="Directory to files used to create processed pose files",
                        default="/Users/will.i.liam/Desktop/final_project/VEE5qqDPVGY/targets.txt",
                        type=str)

    parser.add_argument("--mfcc_exists",
                        default=False,
                        type=bool)
    args = parser.parse_args()
    main(args)
