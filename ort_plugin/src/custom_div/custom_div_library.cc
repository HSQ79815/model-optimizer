#include "custom_div_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cmath>
#include <mutex>
#include <vector>

#include "custom_div_kernel.h"
#include <iostream>

static const char *c_OpDomain = "custom.div";

struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi *ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain *domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi *ort_api_;
};

using OrtCustomOpDomainUniquePtr =
    std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain *domain,
                                            const OrtApi *ort_api) {
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(
      domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue *value) {
    OrtTensorTypeAndShapeInfo *info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct CustomDivKernel {
  CustomDivKernel(OrtApi api, const OrtKernelInfo *info)
      : api_(api), ort_(api_), info_(info) {
    alpha_ = ort_.KernelInfoGetAttribute<float>(info_, "alpha");
  }

  void Compute(OrtKernelContext *context) {
    // Setup inputs
    const OrtValue *input_X = ort_.KernelContext_GetInput(context, 0);
    const OrtValue *input_Y = ort_.KernelContext_GetInput(context, 1);
    const float *X = ort_.GetTensorData<float>(input_X);
    const float *Y = ort_.GetTensorData<float>(input_Y);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtTensorDimensions dimensions2(ort_, input_Y);

    OrtValue *output1 = ort_.KernelContext_GetOutput(
        context, 0, dimensions.data(), dimensions.size());
    float *out1 = ort_.GetTensorMutableData<float>(output1);

    OrtValue *output2 = ort_.KernelContext_GetOutput(
        context, 1, dimensions.data(), dimensions.size());
    float *out2 = ort_.GetTensorMutableData<float>(output2);

    OrtTensorTypeAndShapeInfo *output_info =
        ort_.GetTensorTypeAndShape(output1);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(
        ort_.KernelContext_GetGPUComputeStream(context));

    invoke_custom_div(X, Y, alpha_, out1, out2, size, stream);
  }

private:
  OrtApi api_; // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
  const OrtKernelInfo *info_;
  float alpha_;
};

struct CustomOpDiv : Ort::CustomOpBase<CustomOpDiv, CustomDivKernel> {
  void *CreateKernel(OrtApi api, const OrtKernelInfo *info) const {
    return new CustomDivKernel(api, info);
  };

  const char *GetName() const { return "CustomDiv"; };

  const char *GetExecutionProviderType() const {
    return "CUDAExecutionProvider";
  }

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

} c_CustomOpDiv;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpDiv)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
