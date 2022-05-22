#! /usr/bin/env python3

import polygraph
import polygraphy.backend.onnx
import onnx


def fold_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    return polygraphy.backend.onnx.fold_constants(model)
