
#! /usr/bin/env python3

from cgitb import reset
import tensorrt as trt
import torch
import numpy as np
import os
import argparse
import contextlib
import re
import shlex
import shutil
import subprocess
import sys
import multiprocessing
import time

def TrtType2NumpyType(a):
    if a.name == "BOOL":
        return np.bool_
    elif a.name == "INT8":
        return np.int8
    elif a.name == "INT32":
        return np.int32
    elif a.name == "FLOAT":
        return np.float32
    else:
        return None

class TrtModel:
    
    def __init__(self,engine_path):
        
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.context = self.engine.create_execution_context()
        self.context.set_optimization_profile_async(0, 0)
        self.input_idxs=[]
        self.input_names=[]
        self.input_dtypes=[]
        self.input_np_dtypes=[]
        self.output_idxs=[]
        self.output_names=[]
        self.output_dtypes=[]
        self.output_np_dtypes=[]
        self.get_model_info()
        self.buffers = self.allocate_buffers()           
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def get_model_info(self):
        num_bindings = self.engine.num_bindings

        for idx in range(num_bindings):
            if self.engine.binding_is_input(idx):
                self.input_idxs.append(idx)
                self.input_names.append(self.engine.get_binding_name(idx))
                self.input_dtypes.append(self.engine.get_binding_dtype(idx))
                self.input_np_dtypes.append(TrtType2NumpyType(self.input_dtypes[-1]))
            else:
                self.output_idxs.append(idx)
                self.output_names.append(self.engine.get_binding_name(idx))
                self.output_dtypes.append(self.engine.get_binding_dtype(idx))
                self.output_np_dtypes.append(TrtType2NumpyType(self.output_dtypes[-1]))

        for i in range(len(self.input_idxs)):
            idx = self.input_idxs[i]
            name=self.input_names[i]
            dtype = self.input_dtypes[i]
            shape=self.context.get_binding_shape(idx)
            print({"input name: {0}, input_shape: {1},dtype: {2}".format(name,shape,dtype.name)})

        
    
    def allocate_buffers(self):
        buffers = [None]*(len(self.input_idxs)+len(self.output_idxs))
        return buffers
        
       
            
    def __call__(self,input_datas,device_id,stream_ptr):
        for i in range(len(input_datas)):
            input_idx = self.input_idxs[i]
            input_data=input_datas[i]
            input_ptr = torch.from_numpy(input_data).cuda(device_id).data_ptr()
            self.buffers[input_idx]=input_ptr
            self.context.set_binding_shape(input_idx,input_data.shape)

        output_nps=[]
        for i in range(len(self.output_idxs)):
            output_idx = self.output_idxs[i]
            output_dtype=self.output_np_dtypes[i]
            output_shape=self.context.get_binding_shape(output_idx)
            output_np = np.ones(tuple(output_shape), dtype=output_dtype)
            output_tensor = torch.from_numpy(output_np).cuda(device_id)
            output_ptr = output_tensor.data_ptr()
            output_nps.append(output_tensor)
            self.buffers[output_idx]=output_ptr
        
        self.context.execute_async_v2(self.buffers, 0)
        stream_ptr.synchronize()
        return [x.cpu() for x in output_nps]
        
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--engine_path",required=True,type=str,help="tensorrt engine path")
    parser.add_argument("-i","--input_path",required=True,type=str,help="input numpy file path")
    parser.add_argument("-o","--output_path",required=True,type=str,help="output numpy file path")
    parser.add_argument("--device_id",default=0,type=int,help="gpu device id")    

    args = parser.parse_args()

    trt_engine_path = args.engine_path
    device_id= args.device_id
    torch.cuda.set_device(device_id)
    stream_ptr = torch.cuda.Stream()

    model = TrtModel(trt_engine_path)
    shape = model.engine.get_binding_shape(0)

    data = np.load(args.input_path)
    result = model([data],device_id,stream_ptr)
    np.save(args.output_path,result[0])

    loop = 100
    time_start = time.perf_counter()
    for i in range(100):
        result = model([data],device_id,stream_ptr)
    cost_time = (time.perf_counter() - time_start)
    print("inference cost :{0} ms".format(cost_time*1000/loop))
    