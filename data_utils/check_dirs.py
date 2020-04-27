import os
import sys


def check_input_dir(input_dir):
    if input_dir is None:
        return ""
    elif not (os.path.exists(input_dir) and os.path.isdir(input_dir)):
        print("Invalid input directory, exiting")
        sys.exit(1)

    # otherwise
    if input_dir[-1] != '/':
        input_dir += '/'
    return input_dir

def check_output_dir(output_dir):
    if output_dir is None:
        return check_output_dir(os.getcwd())
    # check if output_dir exists
    elif not(os.path.exists(output_dir) and os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    # after directory has been confirmed or created, make sure output_dir has ending slash
    if output_dir[-1] != '/':
        output_dir += '/'
    return output_dir

def check_mp4(filename):
    if len(filename) < 5 or filename[-4:] != '.mp4':
        return filename + '.mp4'
    return filename

