import onnx
from onnx import ValueInfoProto, NodeProto, ModelProto
from onnx import helper
from onnx import checker

import os
import argparse
import contextlib
import re
import shlex
import shutil
import subprocess
import sys
import multiprocessing
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", required=True,
                    type=str, help="input onnx model")
parser.add_argument("-o", "--output", required=True,
                    type=str, help="output onnx model")
args = parser.parse_args()

model = onnx.load(args.input)

nodes = model.graph.node

# unsequeeze_inputs = {"305": "304",
#                      "313": "312",
#                      "333": "332",
#                      "341": "340",
#                      "388": "387",
#                      "396": "395",
#                      "416": "415",
#                      "424": "423",
#                      "471": "470",
#                      "479": "478",
#                      "499": "498",
#                      "507": "506",
#                      "554": "553",
#                      "562": "561",
#                      "582": "581",
#                      "590": "589",
#                      "637": "636",
#                      "645": "644",
#                      "665": "664",
#                      "673": "672",
#                      "720": "719",
#                      "728": "727",
#                      "748": "747",
#                      "756": "755",
#                      "811": "810",
#                      "831": "830",
#                      "839": "838" }

resize_scales = ["325","353","408","436","491","519","574","602","657","685","740","768","823","851"]

scales_0 = onnx.helper.make_tensor(
    "scales", 1, [4], np.array([1,1,2,2], dtype=np.float32))

initializer = model.graph.initializer

initializer.append(scales_0)

for node in nodes:
    if node.op_type == "Resize":
        input = node.input
        if node.output[0] in resize_scales:
            input[2] = "scales"
            del node.input[3]
        


checker.check_model(model=model)

onnx.save(model, args.output)
