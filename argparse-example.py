"""
Take command line arguments, and print them out
"""

import argparse

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--input', required=True, help='Path to input folder')
ap.add_argument('-o', '--output', help='Path to output folder')
args = vars(ap.parse_args())

print('input: ', args['input'])
print('output: ', args['output'])

