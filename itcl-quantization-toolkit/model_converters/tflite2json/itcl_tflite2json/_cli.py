print("Loading CLI Tool")
import argparse
from itcl_tflite2json import convert


parser = argparse.ArgumentParser(description='TFLite model to Json format')

parser.add_argument('paths', metavar='P', type=int, nargs='2',
                    help='Input and Output model path')

args = parser.parse_args()
paths = args.paths

convert(paths[0], paths[1])