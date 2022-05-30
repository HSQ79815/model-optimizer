#include "customdiv_plugin.h"

template <typename T>
__global__ void CustomDivKernel(const T *src1, const T *src2, const float alpha, T *dest1, T *dest2, const int nElement);

template <>
__global__ void CustomDivKernel<float>(const float *src1, const float *src2, const float alpha, float *dest1, float *dest2, const int nElement)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElement)
        return;
    float s1 = src1[x];
    float s2 = src2[x];
    if (s1 == 0.f)
    {
        dest1[x] = 0.f;
        dest2[x] = 0.f;
    }
    else
    {
        dest1[x] = 1.f;
        dest2[x] = s2 * alpha / s1;
    }
}

template <>
__global__ void CustomDivKernel<__half>(const __half *src1, const __half *src2, const float alpha, __half *dest1, __half *dest2, const int nElement)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElement)
        return;

    float x = __half2float(input[index]);
    float a = fmaf(alpha, x, beta);
    float b = fmaxf(0, fminf(1, a));

    float s1 = __half2float(src1[x]);
    float s2 = __half2float(src2[x]);
    if (s1 == 0.f)
    {
        dest1[x] = __float2half(0.f);
        dest2[x] = __float2half(0.f);
    }
    else
    {
        dest1[x] = __float2half(1.f);
        dest2[x] = __float2half(s2 * alpha / s1);
    }
}

namespace nvinfer1
{
    // class CustomDivPlugin
    CustomDivPlugin::CustomDivPlugin(const std::string &name, const PluginFieldCollection *fc) : name_(name), namespace_("")
    {
        for (int i = 0; i < fc->nbFields; ++i)
        {
            auto &pf = fc->fields[i];
            if (std::string(pf.name) == "alpha")
            {
                alpha_ = *reinterpret_cast<T *>(const_cast<void *>(pf.data));
                break;
            }
        }
    }

    CustomDivPlugin::CustomDivPlugin(const std::string &name, const void *buffer, size_t length) : name_(name), namespace_("")
    {
        const float *buf = reinterpret_cast<const float *>(buffer);
        alpha_ = *buf;
    }

    CustomDivPlugin::~CustomDivPlugin() {}

    IPluginV2DynamicExt *CustomDivPlugin::clone() const noexcept
    {

        auto p = new CustomDivPlugin(name_);
        p->setPluginNamespace(namespace_.c_str());
        return p;
    }

    int32_t CustomDivPlugin::getNbOutputs() const noexcept
    {

        return 2;
    }

    DataType CustomDivPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
    {

        return inputTypes[0];
    }

    DimsExprs CustomDivPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
    {

        return inputs[0];
    }

    bool CustomDivPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
    {

        if (pos == 0) // input0
        {
            return (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) &&
                   (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF);
        }
        else if (pos == 1) // input1
        {
            return (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) &&
                   (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF);
        }
        else if (pos == 2) // output0
        {
            return (inOut[pos].format == inOut[0].format) && (inOut[pos].type == inOut[0].type);
        }
        else if (pos == 3) // output1
        {
            return (inOut[pos].format == inOut[0].format) && (inOut[pos].type == inOut[0].type);
        }
        return false;
    }

    void CustomDivPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
    {
    }

    size_t CustomDivPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
    {

        return 0;
    }

    int32_t CustomDivPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
    {

        int nElement = 1;
        for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
        {
            nElement *= inputDesc[0].dims.d[i];
        }
        dim3 grid(CEIL_DIVIDE(nElement, 1024), 1, 1), block(1024, 1, 1);
        if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
        {
            using dtype = float;
            const dtype *input_0 = static_cast<const dtype *>(inputs[0]);
            const dtype *input_1 = static_cast<const dtype *>(inputs[1]);
            dtype *output_0 = static_cast<dtype *>(outputs[0]);
            dtype *output_1 = static_cast<dtype *>(outputs[1]);
            CustomDivKernel<dtype><<<grid, block, 0, stream>>>(input_0, input_1, alpha_, output_0, output_1, nElement);
        }
        else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
        {
            using dtype = __half;
            const dtype *input_0 = static_cast<const dtype *>(inputs[0]);
            const dtype *input_1 = static_cast<const dtype *>(inputs[1]);
            dtype *output_0 = static_cast<dtype *>(outputs[0]);
            dtype *output_1 = static_cast<dtype *>(outputs[1]);
            CustomDivKernel<dtype><<<grid, block, 0, stream>>>(input_0, input_1, alpha_, output_0, output_1, nElement);
        }
        return 0;
    }

    void CustomDivPlugin::destroy() noexcept
    {
    }

    int32_t CustomDivPlugin::initialize() noexcept
    {

        return 0;
    }

    void CustomDivPlugin::terminate() noexcept
    {
    }

    size_t CustomDivPlugin::getSerializationSize() const noexcept
    {

        return sizeof(float);
    }

    void CustomDivPlugin::serialize(void *buffer) const noexcept
    {
        float *buf = reinterpret_cast<float *>(buffer);
        *buf = alpha_;
    }

    void CustomDivPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
    {

        namespace_ = pluginNamespace;
    }
    const char *CustomDivPlugin::getPluginNamespace() const noexcept
    {

        return namespace_.c_str();
    }

    const char *CustomDivPlugin::getPluginType() const noexcept
    {

        return PLUGIN_NAME;
    }

    const char *CustomDivPlugin::getPluginVersion() const noexcept
    {

        return PLUGIN_VERSION;
    }

    void CustomDivPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
    {
    }

    void CustomDivPlugin::detachFromContext() noexcept
    {
    }

    CustomDivPluginCreator::CustomDivPluginCreator() : namespace_("")
    {
    }

    CustomDivPluginCreator::~CustomDivPluginCreator()
    {
    }

    // 最重要的两个成员函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
    IPluginV2 *CustomDivPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
    {

        return new CustomDivPlugin(name,fc);
    }

    IPluginV2 *CustomDivPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
    {

        return new CustomDivPlugin(name, serialData, serialLength);
    }

    void CustomDivPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
    {

        namespace_ = pluginNamespace;
    }

    const char *CustomDivPluginCreator::getPluginNamespace() const noexcept
    {

        return namespace_.c_str();
    }

    const char *CustomDivPluginCreator::getPluginName() const noexcept
    {

        return PLUGIN_NAME;
    }
    const char *CustomDivPluginCreator::getPluginVersion() const noexcept
    {

        return PLUGIN_VERSION;
    }

    const PluginFieldCollection *CustomDivPluginCreator::getFieldNames() noexcept
    {

        static const std::vector<nvinfer1::PluginField> fields{};
        static nvinfer1::PluginFieldCollection pfc{static_cast<int>(fields.size()), fields.data()};
        return &pfc;
    }

    REGISTER_TENSORRT_PLUGIN(CustomDivPluginCreator);

} // namespace nvinfer1
