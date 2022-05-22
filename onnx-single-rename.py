#! /usr/bin/env python3

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
from onnx import shape_inference
from onnx.tools import update_model_dims
from onnx.tools import net_drawer

from common.convert_to_str import value_infos_to_str,nodes_to_str
from common.rename_input_output import rename_input_output

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_onnx", required=True,
                        type=str, help="input onnx model")
    parser.add_argument("-o", "--output_onnx", required=True,
                        type=str, help="output onnx model")
    return parser.parse_args()

def main(model: onnx.ModelProto, print_info= True)-> onnx.ModelProto:
    decorate = "*"*15
    if print_info: 
        print("before renamed model info")
        print("{0}input{1}\n{2}".format(decorate, decorate,
            value_infos_to_str(model.graph.input)))
        print("{0}output{1}\n{2}".format(decorate, decorate,
            value_infos_to_str(model.graph.output)))
    if (len(model.graph.input) != 1) or (len(model.graph.output) != 1):
        print("input/output size not 1")

    input_rename_dict = {model.graph.input[0].name: "INPUT__0"}
    output_rename_dict = {model.graph.output[0].name: "OUTPUT__0"}
    model = rename_input_output(model, input_rename_dict, output_rename_dict)
    if print_info:
        print("\nafter renamed model info")
        print("{0}input{1}\n{2}".format(decorate, decorate,
            value_infos_to_str(model.graph.input)))
        print("{0}output{1}\n{2}".format(decorate, decorate,
            value_infos_to_str(model.graph.output)))
    return model

    # model = update_model_dims.update_inputs_outputs_dims(model,{"INPUT__0":['b',3,640,640]},{"OUTPUT__0":['b',8400,19]})


if __name__ == "__main__":
    args = make_parse()
    input_model_path = args.input_onnx
    output_model_path = args.output_onnx
    model = onnx.load(input_model_path)
    model = main(model)
    onnx.save(model,output_model_path)
