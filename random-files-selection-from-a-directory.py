"""
Randomly sample n files from a folder.
INPUT: a path, n
OUTPUT: a file contain a list of randomly sampled filenames

Example: python random-files-selection-from-a-directory.py -i '/my/folder' -n5 -o 'my/output/file_name.txt
"""

def sample_from_directory(inpath, n, output_path):
    import os
    import random

    list_fnames = []

    # randomly select from a directory
    random.seed(8)
    list_fnames = random.sample(os.listdir(inpath), n)

    # write to file
    with open(output_path, 'w') as f:
        for item in list_fnames:
            f.writelines(item + '\n')
    f.close()

    return 


if __name__ == "__main__":

    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True, help="Path to a folder")
    parser.add_argument("-n", "--n", required=True, type=int, help="number of items to sample")
    parser.add_argument("-o", "--output_path", required=True, help="Path to write the sampled list to")
    args = vars(parser.parse_args())

    # check for valid folder
    if os.path.exists(args['input_path']):
        
        # classify image into RGB or NIR
        sample_from_directory(args['input_path'], args['n'], args['output_path'] )

    else: 
        print(args['input_path'], ' invalid.')