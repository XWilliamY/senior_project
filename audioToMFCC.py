import librosa # for working with audio in python
import numpy as np
import argparse
from data_utils.check_dirs import check_input_dir, check_output_dir
from data_utils.read_desired_frames import read_desired_frames
import soundfile as sf

def convert(audio_file, output_dir, targets=None):
    """
    time_in_seconds * sample_rate / hop_length = frame in mfcc
    """

    x, sample_rate = librosa.load(audio_file, sr=None)

    total_samples = x.shape[0]
    samples_per_second = sample_rate
    frames_per_sec = 24
    hop_length = 490
    mfccs = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40, hop_length=hop_length)

    save_to = output_dir + audio_file.split('/')[-1][:-4] + "_all_mfccs.npy"
    np.save(save_to, mfccs)
    
    if targets == None:
        # full beginning to ending frame already generated (mfccs)
        return save_to

    # otherwise process targets
    for target_person_id, frame_begin, frame_end in targets:
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
        print(end - initial)
        for i in range(initial, end)[::3]:
            j = i + 3
            print(i, j)
            if i > mfccs.shape[-1]:
                print("initial index exceeds dimensions of mfccs")
                break
            sub_mfccs = mfccs[:, i : j]
            if (sub_mfccs.shape[-1] != 3):
                print("Padding sliced mfcc")
                padded_sub_mfccs = np.zeros([40, 3])
                shape = sub_mfccs.shape
                padded_sub_mfccs[:shape[0], :shape[1]] = sub_mfccs
                sub_mfccs = padded_sub_mfccs

            collect.append(sub_mfccs)

        np_sub_mfccs = np.array(collect)
        np_sub_mfccs = np.transpose(np_sub_mfccs, (1, 0, 2))
        np_sub_mfccs = np_sub_mfccs.reshape(40, -1) # so librosa can read this
        print(np_sub_mfccs.shape)
        save_to = output_dir + audio_file.split(',')[-1][:-4] + '_' + str(frame_begin) + "_" + str(frame_end) + '_mfccs.npy'
        np.save(save_to, np_sub_mfccs)
        return save_to

def main(args):
    input_dir = check_input_dir(args.input_dir)
    output_dir = check_output_dir(args.output_dir)
    filename = input_dir + args.input_audio_filename
    frame_rate = 24
    target_frame_rate = 30
    desired_person_at_frame = read_desired_frames(args.targets, frame_rate, target_frame_rate)

    if not args.mfcc_exists: # if it exists
        convert(filename, output_dir, desired_person_at_frame)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        help="Path to directory containing relevant npy data",                        
                        default='/Users/will.i.liam/Desktop/final_project/VEE5qqDPVGY/audio/',
                        type=str)

    parser.add_argument("--input_audio_filename",
                        help="Name of specific file to preprocess",
                        default='VEE5qqDPVGY.mp3',
                        type=str)
    
    parser.add_argument("--output_dir",
                        help="Path to output directory, default will be same as input_dir",
                        default='/Users/will.i.liam/Desktop/final_project/VEE5qqDPVGY/data/',
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
