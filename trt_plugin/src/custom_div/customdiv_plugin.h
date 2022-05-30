
#pragma once

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <map>
#include <string>
#include <vector>

namespace nvinfer1
{
  class CustomDivPlugin : public IPluginV2DynamicExt
  {
  private:
    const std::string name_;
    std::string namespace_;
    float alpha_;

  public:
    CustomDivPlugin() = delete;
    CustomDivPlugin(const std::string &name,const PluginFieldCollection *fc);
    CustomDivPlugin(const std::string &name, const void *buffer, size_t length);
    ~CustomDivPlugin();

    // Method inherited from IPluginV2
    const char *getPluginType() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void *buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;

    // Method inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index,
                               nvinfer1::DataType const *inputTypes,
                               int32_t nbInputs) const noexcept override;
    void attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas,
                         IGpuAllocator *gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;

    // Method inherited from IPluginV2DynamicExt
    IPluginV2DynamicExt *clone() const noexcept override;
    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs,
                                  int32_t nbInputs,
                                  IExprBuilder &exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs,
                         const DynamicPluginTensorDesc *out,
                         int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs,
                            const PluginTensorDesc *outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(const PluginTensorDesc *inputDesc,
                    const PluginTensorDesc *outputDesc, const void *const *inputs,
                    void *const *outputs, void *workspace,
                    cudaStream_t stream) noexcept override;
  };

  class CustomDivPluginCreator : public IPluginCreator
  {
  private:
    std::string namespace_;

  public:
    CustomDivPluginCreator();
    ~CustomDivPluginCreator();
    const char *getPluginName() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    const PluginFieldCollection *getFieldNames() noexcept override;
    IPluginV2 *createPlugin(const char *name,
                            const PluginFieldCollection *fc) noexcept override;
    IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                 size_t serialLength) noexcept override;
    void setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;
  };

} // namespace nvinfer1
