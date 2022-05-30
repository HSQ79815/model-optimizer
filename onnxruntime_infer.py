#! /usr/bin/env python3
import numpy as np
import onnxruntime as ort
import onnx

import os
import sys
import time
import argparse

if __name__ == "__main__":
    # ort.set_default_logger_severity(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--onnx_path", required=True,
                        type=str, help="onnx path")
    parser.add_argument("-i", "--input_path", required=True,
                        type=str, help="input numpy file path")
    parser.add_argument("-o", "--output_path", required=True,
                        type=str, help="output numpy file path")

    args = parser.parse_args()
    model_name = args.onnx_path
    output_path = args.output_path
    data = np.load(args.input_path)
    ortinput = ort.OrtValue.ortvalue_from_numpy(data, 'cuda', 0)
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opt.register_custom_ops_library("ort_plugin/build/src/libort_custom_op_plugins.so")
    providers = [
        'CUDAExecutionProvider'
    ]

    ses = ort.InferenceSession(
        model_name, sess_options=sess_opt, providers=providers)
    res = ses.run(["OUTPUT__0"], {"INPUT__0": ortinput})
    np.save(output_path,res)

    loop = 100
    time_start = time.perf_counter()
    for i in range(100):
        res = ses.run(["OUTPUT__0"], {"INPUT__0": ortinput})
    cost_time = (time.perf_counter() - time_start)
    print("inference cost :{0} ms".format(cost_time*1000/loop))