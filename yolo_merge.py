import onnx
import onnx.helper
import onnx.checker
import onnx.compose

import os
import argparse
import contextlib
import re
import shlex
import shutil
import subprocess
import sys
import multiprocessing
from common.rename_input_output import rename_input_output


parser = argparse.ArgumentParser()
parser.add_argument("--yolo_c2",default="/Users/haochengqiang/Desktop/yolo_m_c2_renamed.onnx",
                    type=str, help="input onnx model")
parser.add_argument("--yolo_c15",default="/Users/haochengqiang/Desktop/yolo_m_c15_renamed.onnx",
                    type=str, help="input onnx model")
parser.add_argument("-o", "--output", required=True,
                    type=str, help="output onnx model")
args = parser.parse_args()

yolo_c2 = onnx.load(args.yolo_c2)
yolo_c15 = onnx.load(args.yolo_c15)

yolo_c2.graph.output.append(yolo_c2.graph.input[0])

merge_model = onnx.compose.merge_models(yolo_c2,yolo_c15,io_map=[("INPUT__0","INPUT__0")],prefix1="yolo_c2",prefix2="yolo_c15")

merge_model = rename_input_output(merge_model,{"yolo_c2INPUT__0":"INPUT__0"},{"yolo_c2OUTPUT__0":"OUTPUT__0","yolo_c15OUTPUT__0":"OUTPUT__1"})

onnx.save(merge_model,args.output)




