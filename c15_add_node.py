from operator import mod
from statistics import mode
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
import numpy as np


def onnx_res2_pth_res():
    grids = []
    expanded_strides = []
    strides = [8, 16, 32]

    hsizes = [640 // stride for stride in strides]
    wsizes = [640 // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    grids = grids.astype(np.float32)
    allones_0 = np.zeros((1, 8400, 19), dtype=np.float32)
    allones_0[..., :2] = grids
    expanded_strides = np.concatenate(expanded_strides, 1)
    expanded_strides = expanded_strides.astype(np.float32)

    allones_1 = np.ones((1, 8400, 19), dtype=np.float32)
    allones_1[..., :2] = expanded_strides.repeat(2, axis=2)

    return (allones_0, allones_1, expanded_strides.repeat(2, axis=2))


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_onnx", required=True,
                    type=str, help="input onnx model")
parser.add_argument("-o", "--output_onnx", required=True,
                    type=str, help="output onnx model")
args = parser.parse_args()

model = onnx.load(args.input_onnx)

x0, x1, x2 = onnx_res2_pth_res()

grids = onnx.helper.make_tensor("grids", 1, x0.shape, x0)
expanded_strides_0 = onnx.helper.make_tensor(
    "expanded_strides_0", 1, x1.shape, x1)
expanded_strides_1 = onnx.helper.make_tensor(
    "expanded_strides_1", 1, x2.shape, x2)

slice_8400_2_4_start = onnx.helper.make_tensor(
    "slice_8400_2_4_start", 6, [1], np.array([2], dtype=np.int32))
slice_8400_2_4_end = onnx.helper.make_tensor(
    "slice_8400_2_4_end", 6, [1], np.array([4], dtype=np.int32))
slice_8400_2_4_axes = onnx.helper.make_tensor(
    "slice_8400_2_4_axes", 6, [1], np.array([2], dtype=np.int32))

pad_8400_pads = onnx.helper.make_tensor(
    "pad_8400_pads", 7, [6], np.array([0, 0, 2, 0, 0, 15], dtype=np.int64))

slice_start_0 = onnx.helper.make_tensor(
    "slice_start_0", 6, [1], np.array([0], dtype=np.int32))
slice_end_0 = onnx.helper.make_tensor(
    "slice_end_0", 6, [1], np.array([1], dtype=np.int32))

slice_start_1 = onnx.helper.make_tensor(
    "slice_start_1", 6, [1], np.array([1], dtype=np.int32))
slice_end_1 = onnx.helper.make_tensor(
    "slice_end_1", 6, [1], np.array([2], dtype=np.int32))

slice_start_2 = onnx.helper.make_tensor(
    "slice_start_2", 6, [1], np.array([2], dtype=np.int32))
slice_end_2 = onnx.helper.make_tensor(
    "slice_end_2", 6, [1], np.array([3], dtype=np.int32))

slice_start_3 = onnx.helper.make_tensor(
    "slice_start_3", 6, [1], np.array([3], dtype=np.int32))
slice_end_3 = onnx.helper.make_tensor(
    "slice_end_3", 6, [1], np.array([4], dtype=np.int32))

slice_start_4 = onnx.helper.make_tensor(
    "slice_start_4", 6, [1], np.array([4], dtype=np.int32))
slice_end_4 = onnx.helper.make_tensor(
    "slice_end_4", 6, [1], np.array([5], dtype=np.int32))

slice_start_5 = onnx.helper.make_tensor(
    "slice_start_5", 6, [1], np.array([5], dtype=np.int32))
slice_end_5 = onnx.helper.make_tensor(
    "slice_end_5", 6, [1], np.array([20], dtype=np.int32))

slice_axes = onnx.helper.make_tensor(
    "slice_axes", 6, [1], np.array([2], dtype=np.int32))

div_second_0 = onnx.helper.make_tensor(
    "div_second_0", 1, [1, 8400, 1], np.full((8400), 2.0, dtype=np.float32))

initializer = model.graph.initializer

initializer.append(grids)
initializer.append(expanded_strides_0)
initializer.append(expanded_strides_1)

initializer.append(slice_8400_2_4_start)
initializer.append(slice_8400_2_4_end)
initializer.append(slice_8400_2_4_axes)
initializer.append(pad_8400_pads)
# initializer.append(reshape_8400_shapes)

initializer.append(slice_start_0)
initializer.append(slice_end_0)
initializer.append(slice_start_1)
initializer.append(slice_end_1)
initializer.append(slice_start_2)
initializer.append(slice_end_2)
initializer.append(slice_start_3)
initializer.append(slice_end_3)
initializer.append(slice_start_4)
initializer.append(slice_end_4)
initializer.append(slice_start_5)
initializer.append(slice_end_5)
initializer.append(slice_axes)

initializer.append(div_second_0)

for node in model.graph.node:
    if node.name == "Transpose_412":
        node.output[0] = "transpose_412_output"

add_grids = onnx.helper.make_node(
    'Add',
    inputs=['transpose_412_output', "grids"],
    outputs=['add_grids_output'],
    name="Add_grids"
)

mul_expanded_strides = onnx.helper.make_node(
    'Mul',
    inputs=['add_grids_output', "expanded_strides_0"],
    outputs=['mul_expanded_strides'],
    name="Mul_expanded_strides"
)

slice_8400_2_4 = onnx.helper.make_node(
    'Slice',
    inputs=['mul_expanded_strides', 'slice_8400_2_4_start',
            'slice_8400_2_4_end', 'slice_8400_2_4_axes'],
    outputs=['slice_8400_2_4_output'],
    name="slice_8400_2_4"
)

exp_8400_4_6 = onnx.helper.make_node(
    'Exp',
    inputs=['slice_8400_2_4_output'],
    outputs=['exp_8400_4_6_output'],
    name="exp_8400_4_6"
)

mul_expanded_strides_1 = onnx.helper.make_node(
    'Mul',
    inputs=['exp_8400_4_6_output', "expanded_strides_1"],
    outputs=['mul_expanded_strides_1_output'],
    name="mul_expanded_strides_1"
)

sub_8400_4_6 = onnx.helper.make_node(
    'Sub',
    inputs=['mul_expanded_strides_1_output', 'slice_8400_2_4_output'],
    outputs=['sub_8400_4_6_output'],
    name="sub_8400_4_6"
)

pad_8400 = onnx.helper.make_node(
    'Pad',
    inputs=['sub_8400_4_6_output', 'pad_8400_pads'],
    outputs=['pad_8400_output'],
    name="pad_8400"
)

add_8400 = onnx.helper.make_node(
    'Add',
    inputs=['pad_8400_output', 'mul_expanded_strides'],
    outputs=['add_8400_output'],
    name="add_8400"
)


slice_8400_0 = onnx.helper.make_node('Slice', inputs=['add_8400_output', 'slice_start_0', 'slice_end_0', 'slice_axes'], outputs=[
                                     'slice_8400_0_output'], name="slice_8400_0")
slice_8400_1 = onnx.helper.make_node('Slice', inputs=['add_8400_output', 'slice_start_1', 'slice_end_1', 'slice_axes'], outputs=[
                                     'slice_8400_1_output'], name="slice_8400_1")
slice_8400_2 = onnx.helper.make_node('Slice', inputs=['add_8400_output', 'slice_start_2', 'slice_end_2', 'slice_axes'], outputs=[
                                     'slice_8400_2_output'], name="slice_8400_2")
slice_8400_3 = onnx.helper.make_node('Slice', inputs=['add_8400_output', 'slice_start_3', 'slice_end_3', 'slice_axes'], outputs=[
                                     'slice_8400_3_output'], name="slice_8400_3")
slice_8400_4 = onnx.helper.make_node('Slice', inputs=['add_8400_output', 'slice_start_4', 'slice_end_4', 'slice_axes'], outputs=[
                                     'slice_8400_4_output'], name="slice_8400_4")
slice_8400_5 = onnx.helper.make_node('Slice', inputs=['add_8400_output', 'slice_start_5', 'slice_end_5', 'slice_axes'], outputs=[
                                     'slice_8400_5_output'], name="slice_8400_5")
argmax_8400_0 = onnx.helper.make_node('ArgMax', inputs=['slice_8400_5_output'], outputs=[
    'argmax_8400_0_output'], axis=2, keepdims=1, name="argmax_8400_0")
cast_8400_0 = onnx.helper.make_node('Cast', inputs=['argmax_8400_0_output'], outputs=[
    'cast_8400_0_output'], to=1, name="cast_8400_0")

reducemax_8400_0 = onnx.helper.make_node(
    'ReduceMax', inputs=['slice_8400_5_output'], outputs=['reducemax_8400_0_output'], keepdims=1, axes=[2])

mul_8400_0 = onnx.helper.make_node(
    'Mul',
    inputs=['slice_8400_4_output', "reducemax_8400_0_output"],
    outputs=['mul_8400_0_output'],
    name="mul_8400_0"
)  

div_8400_0 = onnx.helper.make_node('Div', inputs=['slice_8400_2_output', 'div_second_0'], outputs=[
                                   'div_8400_0_output'], name="div_8400_0")
div_8400_1 = onnx.helper.make_node('Div', inputs=['slice_8400_3_output', 'div_second_0'], outputs=[
                                   'div_8400_1_output'], name="div_8400_1")

sub_8400_0 = onnx.helper.make_node('Sub', inputs=['slice_8400_0_output', 'div_8400_0_output'], outputs=[
                                   'sub_8400_0_output'], name="sub_8400_0")
sub_8400_1 = onnx.helper.make_node('Sub', inputs=['slice_8400_1_output', 'div_8400_1_output'], outputs=[
                                   'sub_8400_1_output'], name="sub_8400_1")
add_8400_2 = onnx.helper.make_node('Add', inputs=['slice_8400_0_output', 'div_8400_0_output'], outputs=[
                                   'add_8400_2_output'], name="add_8400_2")
add_8400_3 = onnx.helper.make_node('Add', inputs=['slice_8400_1_output', 'div_8400_1_output'], outputs=[
                                   'add_8400_3_output'], name="add_8400_3")

concat_8400_0 = onnx.helper.make_node(
    'Concat',
    inputs=["sub_8400_0_output", "sub_8400_1_output",
            "add_8400_2_output", "add_8400_3_output", "mul_8400_0_output", 'cast_8400_0_output'],
    outputs=['output'],
    axis=2
)

model.graph.output[0].type.tensor_type.shape.dim[2].dim_value=6

model.graph.node.append(add_grids)
model.graph.node.append(mul_expanded_strides)

model.graph.node.append(slice_8400_2_4)
model.graph.node.append(exp_8400_4_6)
model.graph.node.append(mul_expanded_strides_1)
model.graph.node.append(sub_8400_4_6)
model.graph.node.append(pad_8400)
model.graph.node.append(add_8400)
# model.graph.node.append(reshape_8400_6)
model.graph.node.append(slice_8400_0)
model.graph.node.append(slice_8400_1)
model.graph.node.append(slice_8400_2)
model.graph.node.append(slice_8400_3)
model.graph.node.append(slice_8400_4)
model.graph.node.append(slice_8400_5)
model.graph.node.append(argmax_8400_0)
model.graph.node.append(cast_8400_0)
model.graph.node.append(reducemax_8400_0)
model.graph.node.append(mul_8400_0)
model.graph.node.append(div_8400_0)
model.graph.node.append(div_8400_1)
model.graph.node.append(sub_8400_0)
model.graph.node.append(sub_8400_1)
model.graph.node.append(add_8400_2)
model.graph.node.append(add_8400_3)
model.graph.node.append(concat_8400_0)

onnx.checker.check_model(model=model)

onnx.save(model, args.output_onnx)
