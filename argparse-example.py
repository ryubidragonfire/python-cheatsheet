import os
import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True, help="Path to a folder")
    parser.add_argument("-o", "--output_path", required=True, help="Path to write the result to")
    args = vars(parser.parse_args())
    INPUT_PATH = args['input_path']
    OUTPUT_PATH = args['output_path']

    # check for valid folder
    if os.path.exists(INPUT_PATH):
        print('input path: ', INPUT_PATH)
    else: 
        print(INPUT_PATH, ' invalid.')
        sys.exit()

    # create OUTPUT_PATH if not existed
    try:
        os.mkdir(OUTPUT_PATH)
    except OSError:
        print ("Creation of the directory %s failed" % OUTPUT_PATH)
    else:
        print ("Successfully created the directory %s " % OUTPUT_PATH)

