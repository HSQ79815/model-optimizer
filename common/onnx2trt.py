#! /usr/bin/env python3

import tensorrt as trt
import torch
import numpy as np
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import polygraphy
import polygraphy.backend
import polygraphy.backend.onnx
import polygraphy.constants


def covert2trt(model, profile_shapes, fp16, engine_plan_path):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    explicit_precision = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    network = builder.create_network(explicit_batch_flag)

    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(model)

    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        print("parse failed!")
        exit()

    fp32_layer_list = []
    i = 0
    for i in range(network.num_layers):
        layer = network[i]
        print("idx:{0};name: {1}; type: {2};num_inputs: {3};num_outputs: {4};precision: {5};precision_is_set: {6}".format(
            i, layer.name, layer.type, layer.num_inputs, layer.num_outputs, layer.precision, layer.precision_is_set))
    #     # if layer.type != trt.LayerType.SCALE and layer.type != trt.LayerType.RESIZE and layer.type != trt.LayerType.SELECT and layer.type != trt.LayerType.SHAPE and layer.type != trt.LayerType.SLICE and layer.type != trt.LayerType.ELEMENTWISE and (not layer.precision_is_set) and layer.type != trt.LayerType.CONSTANT and layer.type != trt.LayerType.SHUFFLE and layer.type != trt.LayerType.CONCATENATION and layer.type != trt.LayerType.GATHER and layer.type != trt.LayerType.IDENTITY:
        #     fp32_layer_list.append(i)

    # print("len: {0}, idx: {1}".format(len(fp32_layer_list), fp32_layer_list))

    def set_layer_precision(network, idx, precision):
        if idx >= network.num_layers:
            return
        layer = network[idx]
        if layer.precision_is_set:
            print("**************************{} precision has been set".format(idx))
            return
        if layer.type == trt.LayerType.CONSTANT:
            print("**************************{} constant layer".format(idx))
            return
        if layer.type == trt.LayerType.SHUFFLE:
            print("**************************{} SHUFFLE layer".format(idx))
            return
        if layer.type == trt.LayerType.CONCATENATION:
            print("**************************{} CONCATENATION layer".format(idx))
            return
        if layer.type == trt.LayerType.GATHER:
            print("**************************{} GATHER layer".format(idx))
            return
        if layer.type == trt.LayerType.IDENTITY:
            print("**************************{} IDENTITY layer".format(idx))
            return
        layer.reset_precision()
        layer.precision = precision
        for i in range(layer.num_outputs):
            if layer.get_output_type(i) != trt.int32 and layer.get_output_type(i) != trt.bool:
                layer.set_output_type(i, precision)

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 33
    profile = builder.create_optimization_profile()
    for i in range(len(profile_shapes)):
        input_name = "INPUT__" + str(i)
        min_opt_max_shape = profile_shapes[input_name]
        min_shape = min_opt_max_shape[0]
        opt_shape = min_opt_max_shape[1]
        max_shape = min_opt_max_shape[2]
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.DEBUG)
    config.set_flag(trt.BuilderFlag.DIRECT_IO)
    config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
    # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # def remove_items(a, b):
    #     for idx in b:
    #         if idx in a:
    #             a.remove(idx)

    # fp32_list = fp32_layer_list

    # for idx in fp32_list:
    #     print("{0} set precision".format(idx))
    #     set_layer_precision(network, idx, trt.float32)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_plan_path, "wb") as f:
        f.write(serialized_engine)
