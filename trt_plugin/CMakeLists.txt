cmake_minimum_required (VERSION 3.12.0)
project(custom_plugin  LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
if(DISABLE_DEPRECATION_WARNING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
endif(DISABLE_DEPRECATION_WARNING)
    
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-overloaded-virtual")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-overloaded-virtual --expt-relaxed-constexpr")


cmake_policy(SET CMP0104 OLD)
set(CMAKE_CUDA_FLAGS "--compiler-options -fPIC -Wno-deprecated-gpu-targets")
foreach(arch 50 52 53 60 61 62 70 72 75 80)
    set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${arch},code=sm_${arch}")
endforeach()

option(GPU_A10 "enable A10" OFF)

if(GPU_A10)
    set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")
endif()

option(ENABLE_TEST "enable build test" OFF)


set(CUDA_DIR "/usr/local/cuda")
set(TENSORRT_HOME "/opt/tensorrt/")


include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CUDA_DIR}/include)
include_directories(/usr/local/include)
include_directories(${TENSORRT_HOME}/include)


link_directories(${CUDA_DIR}/lib64)
link_directories(${CUDA_DIR}/targets/x86_64-linux/lib)
link_directories(${TENSORRT_HOME}/lib)
link_directories(/usr/lib/x86_64-linux-gnu/)
 
set(EXTRALIBS ${EXTRALIBS} rt pthread ${CMAKE_DL_LIBS})
set(EXTRALIBS ${EXTRALIBS} cuda cudnn cublas cudart nvToolsExt )
set(EXTRALIBS ${EXTRALIBS} nvinfer  -Wl,--no-as-needed nvinfer_plugin -Wl,--as-needed)

add_subdirectory(src)
    
