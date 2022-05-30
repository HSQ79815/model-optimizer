cmake_minimum_required (VERSION 3.12.0)
project(ort_custom_op  LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/bin")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type -Wno-error=deprecated-declarations -Wno-deprecated-declarations  " )
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type -Wno-error=deprecated-declarations -Wno-deprecated-declarations ")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type -Wno-error=deprecated-declarations -Wno-deprecated-declarations ")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type -Wno-error=deprecated-declarations -Wno-deprecated-declarations ")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_CUDA_FLAGS "--compiler-options -fPIC -Wno-deprecated-gpu-targets")
foreach(arch 50 52 53 60 61 62 70 72 75 80)
    set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${arch},code=sm_${arch}")
endforeach()

option(ENABLE_TEST "enable build test" OFF)
option(ENABLE_OPENMP "enble openmp" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
    message("OpenMP FOUND: ${OpenMP_FOUND}")
endif()

find_package(Threads REQUIRED)

set(CUDA_DIR "/usr/local/cuda")
set(ONNXRUNTIME_HOME "/opt/onnxruntime/")

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CUDA_DIR}/include)
include_directories(/usr/local/include)
include_directories(${ONNXRUNTIME_HOME}/include)
include_directories(${ONNXRUNTIME_HOME}/include/onnxruntime)
include_directories(${ONNXRUNTIME_HOME}/include/onnxruntime/core)
include_directories(${ONNXRUNTIME_HOME}/include/onnxruntime/core/providers)
include_directories(${ONNXRUNTIME_HOME}/include/onnxruntime/core/session)

link_directories(${CUDA_DIR}/lib64)
link_directories(${CUDA_DIR}/targets/x86_64-linux/lib)
link_directories(${ONNXRUNTIME_HOME}/lib)
link_directories(/usr/local/cuda/lib64/stubs/)
 
set(EXTRALIBS ${EXTRALIBS} rt pthread  cuda cudnn cublas cudart nvToolsExt ${CMAKE_DL_LIBS})
set(EXTRALIBS ${EXTRALIBS}
    ${ONNXRUNTIME_HOME}/lib/libcustom_op_library.so
    ${ONNXRUNTIME_HOME}/lib/libonnxruntime_providers_shared.so
    ${ONNXRUNTIME_HOME}/lib/libonnxruntime_providers_tensorrt.so
    ${ONNXRUNTIME_HOME}/lib/libonnxruntime_providers_cuda.so
    ${ONNXRUNTIME_HOME}/lib/libonnxruntime.so
)

add_subdirectory(src)

