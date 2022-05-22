#! /usr/bin/env python3

import onnx
from onnx import ValueInfoProto, NodeProto, ModelProto


def rename_input_output(model: ModelProto, input_rename_dict: dict, output_rename_dict: dict):
    for input in model.graph.input:
        old_name = input.name
        if old_name in input_rename_dict:
            input.name = input_rename_dict[old_name]
    for output in model.graph.output:
        old_name = output.name
        if old_name in output_rename_dict:
            output.name = output_rename_dict[old_name]

    for node in model.graph.node:
        inputs = node.input
        outputs = node.output
        for i in range(len(inputs)):
            old_name = inputs[i]
            if old_name in input_rename_dict:
                inputs[i] = input_rename_dict[old_name]
        for i in range(len(outputs)):
            old_name = outputs[i]
            if old_name in output_rename_dict:
                outputs[i] = output_rename_dict[old_name]

    onnx.checker.check_model(model)
    return model
