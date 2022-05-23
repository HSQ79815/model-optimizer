from audioop import add
import onnx
from onnx import TensorProto,ValueInfoProto,NodeProto,GraphProto,ModelProto
import onnx.helper
import onnx.compose

# import sclblonnx as so


def create_models():
    input_0 = onnx.helper.make_tensor_value_info("input_0",1,[2,2])
    output_0 = onnx.helper.make_tensor_value_info("output_0",1,[2,2])
    add_0 = onnx.helper.make_node("Add",["input_0","input_0"],["output_0"],"add_0")
    graph_0 = onnx.helper.make_graph([add_0],"graph_0",[input_0],[output_0,input_0])
    model_0 = onnx.helper.make_model(graph_0)

    input_1 = onnx.helper.make_tensor_value_info("input_1",1,[2,2])
    output_1 = onnx.helper.make_tensor_value_info("output_1",1,[2,2])
    add_1 = onnx.helper.make_node("Add",["input_1","input_1"],["output_1"],"add_1")
    graph_1 = onnx.helper.make_graph([add_1],"graph_1",[input_1],[output_1])
    model_1 = onnx.helper.make_model(graph_1)
    return (model_0,model_1)



if __name__ == "__main__":
    model_0,model_1 = create_models()

    model_2 = onnx.compose.merge_models(model_0,model_1,io_map=[("input_0","input_1")])
    # model_2 = onnx.compose.merge_models(model_0,model_1,io_map=[])
    onnx.save(model_2,"./model2.onnx")