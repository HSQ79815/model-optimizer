
from cgi import print_arguments
import os
import argparse
import contextlib
import re
import shlex
import shutil
from statistics import mode
import subprocess
import sys
import onnx
from common.fold_constatns import fold_constants


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_onnx", required=True,
                        type=str, help="input onnx model")
    parser.add_argument("-o", "--output_onnx", required=True,
                        type=str, help="output onnx model")
    return parser.parse_args()


if __name__ == "__main__":
    args = make_parse()
    input_model_path = args.input_onnx
    output_model_path = args.output_onnx
    model = onnx.load(input_model_path)
    model = fold_constants(model)
    onnx.save(model, output_model_path)
