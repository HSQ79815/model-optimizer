#! /usr/bin/env python3

import onnx
from onnx import ValueInfoProto, NodeProto

datatype_list = ["UNDEFINED", "FLOAT", "UINT8", "INT8", "UINT16",
                 "INT16", "INT32", "INT64", "STRING", "BOOL", "DOUBLE",
                 "FLOAT16", "UINT32", "UINT64", "COMPLEX64", "COMPLEX128",
                 "BFLOAT16"]

def to_datatype(elem_type: int):
    return datatype_list[elem_type]

def value_info_to_str(value_info: ValueInfoProto):
    name = value_info.name
    tensor_type = value_info.type.tensor_type
    ss = "name: " + name + ", dtype: " + \
        to_datatype(tensor_type.elem_type) + ", shape: ["
    first = True
    for d in tensor_type.shape.dim:
        if first:
            first = False
        else:
            ss += ", "
        if d.HasField("dim_value"):
            ss += str(d.dim_value)
        elif d.HasField("dim_param"):
            ss += d.dim_param
    ss = ss + ']'
    return ss

def value_infos_to_str(value_infos):
    first = True
    ss = ""
    for value_info in value_infos:
        if first:
            first = False
        else:
            ss += "\n"
        ss += value_info_to_str(value_info)
    return ss

def node_to_str(node: onnx.NodeProto):
    ss = "op_type: " + node.op_type + ","
    ss += "name: " + node.name + ","
    ss += "input: " + str(node.input) + ","
    ss += "output: " + str(node.output) + ","
    ss += "attribute: " + str(node.attribute)
    return ss


def nodes_to_str(nodes):
    first = True
    ss = ""
    for node in nodes:
        if first:
            first = False
        else:
            ss += "\n"
        ss += node_to_str(node)
    return ss









