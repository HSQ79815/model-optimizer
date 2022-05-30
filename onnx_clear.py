
from cgi import print_arguments
import enum
from gettext import find
import os
import argparse
import contextlib
import re
import shlex
import shutil
from statistics import mode
import subprocess
import sys
from xml.dom.minicompat import NodeList
import onnx
# import onnx_graphsurgeon as gs
import onnx.helper
import onnx.checker


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_onnx", required=True,
                        type=str, help="input onnx model")
    parser.add_argument("-o", "--output_onnx", required=True,
                        type=str, help="output onnx model")
    return parser.parse_args()


def remove_nodes(model):
    remove_names = ["174", "175",  "186", "199", "200", "211",
                    "224", "225", "236", "249", "250", "261", "274", "275", "286",
                    "357", "358", "369", "440", "441", "452", "523", "524", "535",
                    "606", "607", "618", "689", "690", "701", "772", "773", "784"]
    remove_items = []
    for node in model.graph.node:
        if node.input[0] in remove_names:
            remove_items.append(node)

    for remove_node in remove_items:
        model.graph.node.remove(remove_node)


def adds(model):
    nodes = model.graph.node
    for node in nodes:
        if node.op_type == "Mul":
            if node.input[0] == "183":
                node.input[0] = "174"
            elif node.input[0] == "208":
                node.input[0] = "199"
            elif node.input[0] == "233":
                node.input[0] = "224"
            elif node.input[0] == "258":
                node.input[0] = "249"
            elif node.input[0] == "283":
                node.input[0] = "274"
            elif node.input[0] == "366":
                node.input[0] = "357"
            elif node.input[0] == "449":
                node.input[0] = "440"
            elif node.input[0] == "532":
                node.input[0] = "523"
            elif node.input[0] == "615":
                node.input[0] = "606"
            elif node.input[0] == "698":
                node.input[0] = "689"
            elif node.input[0] == "781":
                node.input[0] = "772"
        elif node.op_type == "Where":
            if node.input[2] == "187":
                node.input[2] = "186"
            elif node.input[2] == "212":
                node.input[2] = "211"
            elif node.input[2] == "237":
                node.input[2] = "236"
            elif node.input[2] == "262":
                node.input[2] = "261"
            elif node.input[2] == "287":
                node.input[2] = "286"
            elif node.input[2] == "370":
                node.input[2] = "369"
            elif node.input[2] == "453":
                node.input[2] = "452"
            elif node.input[2] == "536":
                node.input[2] = "535"
            elif node.input[2] == "619":
                node.input[2] = "618"
            elif node.input[2] == "702":
                node.input[2] = "701"
            elif node.input[2] == "785":
                node.input[2] = "784"


def remove_cast(model):
    nodes = model.graph.node
    cast_nodes = []
    modify_map = {}
    for node in nodes:
        if node.op_type == "Cast":
            cast_nodes.append(node)
            modify_map[node.output[0]] = node.input[0]
    for node in cast_nodes:
        nodes.remove(node)
    for node in nodes:
        if node.input[0] in modify_map.keys():
            node.input[0] = modify_map[node.input[0]]


def add_node_name(model):
    nodes = model.graph.node
    for i, node in enumerate(nodes):
        node.name = str(i)


def find_node(nodes, name):
    for node in nodes:
        if node.name == str(name):
            return node


def remove_other(model):
    nodes = model.graph.node
    start = [51, 67, 83, 99, 115, 135, 155, 175, 195, 215, 235]
    rm_nodes = []
    for s in start:
        rm_nodes.append(find_node(nodes, s))
        rm_nodes.append(find_node(nodes, s+2))
        rm_nodes.append(find_node(nodes, s+3))
        rm_nodes.append(find_node(nodes, s+4))
        rm_nodes.append(find_node(nodes, s+6))
        rm_nodes.append(find_node(nodes, s+7))
        rm_nodes.append(find_node(nodes, s+8))
        rm_nodes.append(find_node(nodes, s+10))

    for node in rm_nodes:
        nodes.remove(node)
    custom_list = [[("177", "174"), ("195", "190"), 3200.],
                   [("202", "199"), ("220", "215"), 2304.],
                   [("227", "224"), ("245", "240"), 4608.],
                   [("252", "249"), ("270", "265"), 4608.],
                   [("277", "274"), ("295", "290"), 4608.],
                   [("360", "357"), ("378", "373"), 9216.],
                   [("443", "440"), ("461", "456"), 9216.],
                   [("526", "523"), ("544", "539"), 9216.],
                   [("609", "606"), ("627", "622"), 6912.],
                   [("692", "689"), ("710", "705"), 3456.],
                   [("775", "772"), ("793", "788"), 1728.],
                   ]
    for i, aa in enumerate(custom_list):
        input, output, alpha  = aa
        custom_node = onnx.helper.make_node(
            'CustomDiv',
            inputs=input,
            outputs=output,
            name="Custom" + str(i),
            domain="custom.div",
            alpha=alpha
        )
        nodes.append(custom_node)


if __name__ == "__main__":
    args = make_parse()
    input_model_path = args.input_onnx
    output_model_path = args.output_onnx
    model = onnx.load(input_model_path)

    # gs_graph = gs.import_onnx(model)
    # gs_graph.cleanup(remove_unused_node_outputs=True, recurse_subgraphs=True)
    # model = gs.export_onnx(gs_graph)

    # remove_nodes(model)
    # adds(model)
    # add_node_name(model)
    # remove_cast(model)
    remove_other(model)
    # gs_graph = gs.import_onnx(model)

    # gs_graph.cleanup(remove_unused_node_outputs=True, recurse_subgraphs=True)
    # model = gs.export_onnx(gs_graph)
    # onnx.checker.check_model(model=model)
    onnx.save(model, output_model_path)
