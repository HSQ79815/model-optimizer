#include "hardswish_plugin.h"

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define CEIL_TO(X, Y)     (CEIL_DIVIDE(X, Y) * (Y))

namespace
{
static const char *PLUGIN_NAME {"HardSwish"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

template <typename T>
__global__ void HardSwishKernel(const T *input, T *output, const int nElement);

template <>
__global__ void HardSwishKernel<float>(const float *input, float *output, const int nElement)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElement)
        return;
    const float alpha = 0.1666667f;
    const float beta = 0.5f;

    float x = input[index];
    float a = fmaf(alpha, x, beta);
    float b = fmaxf(0.0f, fminf(1.0f, a));

    output[index] = x * b;
}

template <>
__global__ void HardSwishKernel<__half>(const __half *input, __half *output, const int nElement)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElement)
        return;
    const float alpha = 0.1666667f;
    const float beta = 0.5f;

    float x = __half2float(input[index]);
    float a = fmaf(alpha, x, beta);
    float b = fmaxf(0, fminf(1, a));

    output[index] = __float2half(x * b);
}

namespace nvinfer1
{
    // class HardSwishPlugin
    HardSwishPlugin::HardSwishPlugin(const std::string &name) : name_(name), namespace_("")
    {
    }

    HardSwishPlugin::HardSwishPlugin(const std::string &name, const void *buffer, size_t length) {}

    HardSwishPlugin::~HardSwishPlugin() {}

    IPluginV2DynamicExt *HardSwishPlugin::clone() const noexcept
    {

        auto p = new HardSwishPlugin(name_);
        p->setPluginNamespace(namespace_.c_str());
        return p;
    }

    int32_t HardSwishPlugin::getNbOutputs() const noexcept
    {

        return 1;
    }

    DataType HardSwishPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
    {

        return inputTypes[0];
    }

    DimsExprs HardSwishPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
    {

        return inputs[0];
    }

    bool HardSwishPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
    {

        if (pos == 0) // input0
        {
            return (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) &&
                   (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF);
        }
        else if (pos == 1) // output
        {
            return (inOut[pos].format == inOut[0].format) && (inOut[pos].type == inOut[0].type);
        }
        return false;
    }

    void HardSwishPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
    {
    }

    size_t HardSwishPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
    {

        return 0;
    }

    int32_t HardSwishPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
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
            const dtype *input = static_cast<const dtype *>(inputs[0]);
            dtype *output = static_cast<dtype *>(outputs[0]);
            HardSwishKernel<dtype><<<grid, block, 0, stream>>>(input, output, nElement);
        }
        else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
        {
            using dtype = __half;
            const dtype *input = static_cast<const dtype *>(inputs[0]);
            dtype *output = static_cast<dtype *>(outputs[0]);
            HardSwishKernel<dtype><<<grid, block, 0, stream>>>(input, output, nElement);
        }
        return 0;
    }

    void HardSwishPlugin::destroy() noexcept
    {
    }

    int32_t HardSwishPlugin::initialize() noexcept
    {

        return 0;
    }

    void HardSwishPlugin::terminate() noexcept
    {
    }

    size_t HardSwishPlugin::getSerializationSize() const noexcept
    {

        return 0;
    }

    void HardSwishPlugin::serialize(void *buffer) const noexcept
    {
    }

    void HardSwishPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
    {

        namespace_ = pluginNamespace;
    }
    const char *HardSwishPlugin::getPluginNamespace() const noexcept
    {

        return namespace_.c_str();
    }

    const char *HardSwishPlugin::getPluginType() const noexcept
    {

        return PLUGIN_NAME;
    }

    const char *HardSwishPlugin::getPluginVersion() const noexcept
    {

        return PLUGIN_VERSION;
    }

    void HardSwishPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
    {
    }

    void HardSwishPlugin::detachFromContext() noexcept
    {
    }

    HardSwishPluginCreator::HardSwishPluginCreator() : namespace_("")
    {
    }

    HardSwishPluginCreator::~HardSwishPluginCreator()
    {
    }

    // 最重要的两个成员函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
    IPluginV2 *HardSwishPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
    {

        return new HardSwishPlugin(name);
    }

    IPluginV2 *HardSwishPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
    {

        return new HardSwishPlugin(name, serialData, serialLength);
    }

    void HardSwishPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
    {

        namespace_ = pluginNamespace;
    }

    const char *HardSwishPluginCreator::getPluginNamespace() const noexcept
    {

        return namespace_.c_str();
    }

    const char *HardSwishPluginCreator::getPluginName() const noexcept
    {

        return PLUGIN_NAME;
    }
    const char *HardSwishPluginCreator::getPluginVersion() const noexcept
    {

        return PLUGIN_VERSION;
    }

    const PluginFieldCollection *HardSwishPluginCreator::getFieldNames() noexcept
    {

        static const std::vector<nvinfer1::PluginField> fields{};
        static nvinfer1::PluginFieldCollection pfc{static_cast<int>(fields.size()), fields.data()};
        return &pfc;
    }

    REGISTER_TENSORRT_PLUGIN(HardSwishPluginCreator);

} // namespace nvinfer1
