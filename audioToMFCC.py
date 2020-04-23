import librosa # for working with audio in python
import numpy as np
import argparse
from data_utils.check_dirs import check_input_dir, check_output_dir
from data_utils.read_desired_frames import read_desired_frames
import soundfile as sf

def convert(input_dir, output_dir, audio_file, targets=None):
    """
    time_in_seconds * sample_rate / hop_length = frame in mfcc
    """

    x, sample_rate = librosa.load(audio_file, sr=None)

    total_samples = x.shape[0]
    samples_per_second = sample_rate
    frames_per_sec = 24
    hop_length = 490
    mfccs = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40, hop_length=hop_length)
    print(mfccs.shape)

    np.save(output_dir + audio_file.split('/')[-1][:-4] + "_all_mfccs.npy", mfccs)
    
    if targets == None:
        return

    # otherwise process targets
    for target_person_id, frame_begin, frame_end in targets:
        collect = []
        """
        targets.txt has video frames corresponding to pre-processed fps
        We have to scale proportionally to 30 fps
        """
        adjust_begin = frame_begin / 24 * 30 # TODO: Don't hard code this
        adjust_end = frame_end / 24 * 30 # video frame

        print(adjust_begin, adjust_end)
        if adjust_begin < 0:
            adjust_begin = 0 
        if adjust_end > total_samples / sample_rate * 30: # last frame
            adjust_end = total_samples / sample_rate * 30 

        # convert from video frames into mfcc frames 
        initial = round(adjust_begin / 30 * sample_rate / hop_length) # assume 512 as hop length
        end = (adjust_end / 30 * sample_rate / hop_length)
        print(initial, end)

        '''
        for i in range(round(initial), round(end))[::9]:
            j = i + 9
            print(i, j)
            if i > mfccs.shape[-1]:
                print("initial index exceeds dimensions of mfccs")
                break
            sub_mfccs = mfccs[:, i : j]
            if (sub_mfccs.shape[-1] != 9):
                print("Padding sliced mfcc")
                padded_sub_mfccs = np.zeros([40, 9])
                shape = sub_mfccs.shape
                padded_sub_mfccs[:shape[0], :shape[1]] = sub_mfccs
                sub_mfccs = padded_sub_mfccs

            collect.append(sub_mfccs)

        np_sub_mfccs = np.array(collect)
        np_sub_mfccs = np.transpose(np_sub_mfccs, (1, 0, 2))
        np_sub_mfccs = np_sub_mfccs.reshape(40, -1)

        np.save(output_dir + audio_file.split('/')[-1][:-4] + '_' + str(frame_begin) + "_" + str(frame_end) + '_mfccs.npy', np_sub_mfccs)
        reconstructed = librosa.feature.inverse.mfcc_to_audio(np_sub_mfccs)
        sf.write(output_dir + '_' + str(frame_begin) + "_" + str(frame_end) + '_mfccs.wav', reconstructed, sample_rate)
    '''

def main(args):
    input_dir = check_input_dir(args.input_dir)
    output_dir = check_output_dir(args.output_dir)
    filename = input_dir + args.input_audio_filename
    desired_person_at_frame = read_desired_frames(args.targets)

    if not args.mfcc_exists: # if it exists
        convert(input_dir, output_dir, filename, desired_person_at_frame)

    
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
