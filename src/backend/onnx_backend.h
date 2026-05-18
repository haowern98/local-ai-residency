#ifndef MOSAICVRAM_SRC_BACKEND_ONNX_BACKEND_H_
#define MOSAICVRAM_SRC_BACKEND_ONNX_BACKEND_H_

#ifdef MOSAICVRAM_ENABLE_ONNX

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend/backend.h"

namespace mosaicvram {

/**
 * Selects whether ONNX inputs and outputs are staged on CPU or CUDA memory.
 */
enum class OnnxIoMode {
  kCpu,
  kCuda,
};

/**
 * Selects the ONNX Runtime execution provider.
 */
enum class OnnxProvider {
  kCpu,
  kCuda,
};

/**
 * Selects how synthetic tensor inputs are initialized for smoke runs.
 */
enum class OnnxInputMode {
  kZero,
  kOne,
  kRamp,
  kRandom,
};

/**
 * Explicit shape override for dynamic ONNX model inputs.
 */
struct OnnxShapeOverride {
  std::string name;
  std::vector<int64_t> shape;
};

/**
 * Configuration for the lower-level ONNX tensor backend.
 */
struct OnnxBackendOptions {
  std::string model_path;
  OnnxIoMode io_mode = OnnxIoMode::kCpu;
  OnnxProvider provider = OnnxProvider::kCuda;
  OnnxInputMode input_mode = OnnxInputMode::kRamp;
  std::uint32_t seed = 123;
  int device_index = 0;
  std::vector<OnnxShapeOverride> shape_overrides;
};

/**
 * ONNX Runtime tensor backend used to validate lifecycle and boundary tensors.
 *
 * This backend is not a chat adapter. It exercises ONNX Runtime loading,
 * execution, and explicit CPU/CUDA input-output buffer ownership.
 */
class OnnxBackend : public Backend {
 public:
  explicit OnnxBackend(OnnxBackendOptions options);
  ~OnnxBackend() override;

  OnnxBackend(const OnnxBackend&) = delete;
  OnnxBackend& operator=(const OnnxBackend&) = delete;

  BackendInfo info() const override;
  BackendRunResult Load() override;
  BackendRunResult RunOnce() override;
  void Unload() override;
  std::vector<BackendMetric> Metrics() const override;

 private:
  struct TensorBuffer {
    std::string name;
    std::vector<int64_t> shape;
    ONNXTensorElementDataType element_type =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::size_t element_count = 0;
    std::vector<float> host_data;
    float* device_data = nullptr;
  };

  BackendRunResult ValidateLoaded() const;
  BackendRunResult DiscoverModelIo();
  BackendRunResult DiscoverInput(std::size_t index,
                                 Ort::AllocatorWithDefaultOptions* allocator);
  BackendRunResult DiscoverOutput(std::size_t index,
                                  Ort::AllocatorWithDefaultOptions* allocator);
  BackendRunResult ResolveShape(TensorBuffer* tensor) const;
  BackendRunResult FillInput(TensorBuffer* tensor, std::uint32_t seed) const;
  BackendRunResult AllocateCudaBuffers();
  void FreeCudaBuffers();
  BackendRunResult RunCpuOnce();
  BackendRunResult RunCudaOnce();
  BackendRunResult ValidateOutputs(
      const std::vector<TensorBuffer>& outputs) const;
  const OnnxShapeOverride* FindShapeOverride(const std::string& name) const;

  OnnxBackendOptions options_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<TensorBuffer> inputs_;
  std::vector<TensorBuffer> outputs_;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_ONNX

#endif  // MOSAICVRAM_SRC_BACKEND_ONNX_BACKEND_H_
