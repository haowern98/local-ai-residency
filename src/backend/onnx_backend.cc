#include "backend/onnx_backend.h"

#ifdef MOSAICVRAM_ENABLE_ONNX

#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cuda/cuda_error.h"

namespace mosaicvram {

namespace {

std::wstring ToWideString(const std::string& text) {
  return std::wstring(text.begin(), text.end());
}

BackendRunResult OkResult() { return BackendRunResult{true, ""}; }

BackendRunResult ErrorResult(const std::string& message) {
  return BackendRunResult{false, message};
}

const char* ElementTypeName(ONNXTensorElementDataType element_type) {
  switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return "uint64";
    default:
      return "unsupported";
  }
}

BackendRunResult ComputeElementCount(const std::vector<int64_t>& shape,
                                     std::size_t* element_count) {
  std::size_t count = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0) {
      return ErrorResult("tensor shape contains unresolved dynamic dimension");
    }
    const std::uint64_t unsigned_dim = static_cast<std::uint64_t>(dim);
    if (unsigned_dim > std::numeric_limits<std::size_t>::max()) {
      return ErrorResult("tensor dimension is too large");
    }
    const std::size_t size_dim = static_cast<std::size_t>(unsigned_dim);
    if (count > std::numeric_limits<std::size_t>::max() / size_dim) {
      return ErrorResult("tensor element count overflow");
    }
    count *= size_dim;
  }
  *element_count = count;
  return OkResult();
}

bool IsFiniteVector(const std::vector<float>& values) {
  for (const float value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

std::string FormatDouble(double value) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(6) << value;
  return stream.str();
}

std::string FormatShape(const std::vector<int64_t>& shape) {
  std::ostringstream stream;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      stream << "x";
    }
    stream << shape[i];
  }
  return stream.str();
}

}  // namespace

OnnxBackend::OnnxBackend(OnnxBackendOptions options)
    : options_(std::move(options)) {}

OnnxBackend::~OnnxBackend() { Unload(); }

BackendInfo OnnxBackend::info() const {
  const MemoryControlLevel memory_control_level =
      options_.io_mode == OnnxIoMode::kCuda
          ? MemoryControlLevel::kBoundaryTensor
          : MemoryControlLevel::kLifecycleOnly;
  return BackendInfo{"onnx", memory_control_level};
}

BackendRunResult OnnxBackend::Load() {
  try {
    if (options_.io_mode == OnnxIoMode::kCuda ||
        options_.provider == OnnxProvider::kCuda) {
      MOSAICVRAM_CUDA_CHECK(cudaSetDevice(options_.device_index),
                            "set ONNX CUDA device");
    }

    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                      "mosaicvram-onnx");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (options_.provider == OnnxProvider::kCuda) {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = options_.device_index;
      session_options_->AppendExecutionProvider_CUDA(cuda_options);
    }

    const std::wstring model_path = ToWideString(options_.model_path);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(),
                                              *session_options_);

    BackendRunResult result = DiscoverModelIo();
    if (!result.ok) {
      Unload();
      return result;
    }
    if (options_.io_mode == OnnxIoMode::kCuda) {
      return AllocateCudaBuffers();
    }
    return OkResult();
  } catch (const std::exception& error) {
    Unload();
    return ErrorResult(error.what());
  }
}

BackendRunResult OnnxBackend::RunOnce() {
  BackendRunResult loaded = ValidateLoaded();
  if (!loaded.ok) {
    return loaded;
  }

  if (options_.io_mode == OnnxIoMode::kCuda) {
    return RunCudaOnce();
  }
  return RunCpuOnce();
}

std::vector<BackendMetric> OnnxBackend::Metrics() const {
  std::vector<BackendMetric> metrics;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;

  for (const TensorBuffer& input : inputs_) {
    input_bytes += input.host_data.size() * sizeof(float);
  }
  for (const TensorBuffer& output : outputs_) {
    output_bytes += output.host_data.size() * sizeof(float);
  }

  metrics.push_back({"onnx_input_count", std::to_string(inputs_.size())});
  metrics.push_back({"onnx_output_count", std::to_string(outputs_.size())});
  metrics.push_back({"onnx_input_bytes", std::to_string(input_bytes)});
  metrics.push_back({"onnx_output_bytes", std::to_string(output_bytes)});

  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    const TensorBuffer& input = inputs_[i];
    const std::string prefix = "onnx_input" + std::to_string(i);
    metrics.push_back({prefix + "_name", input.name});
    metrics.push_back({prefix + "_shape", FormatShape(input.shape)});
    metrics.push_back(
        {prefix + "_elements", std::to_string(input.element_count)});
  }

  for (std::size_t i = 0; i < outputs_.size(); ++i) {
    const TensorBuffer& output = outputs_[i];
    const std::string prefix = "onnx_output" + std::to_string(i);
    metrics.push_back({prefix + "_name", output.name});
    metrics.push_back({prefix + "_shape", FormatShape(output.shape)});
    metrics.push_back(
        {prefix + "_elements", std::to_string(output.element_count)});
    if (!output.host_data.empty()) {
      float min_value = output.host_data.front();
      float max_value = output.host_data.front();
      double checksum = 0.0;
      for (const float value : output.host_data) {
        if (value < min_value) {
          min_value = value;
        }
        if (value > max_value) {
          max_value = value;
        }
        checksum += static_cast<double>(value);
      }
      metrics.push_back(
          {prefix + "_min", FormatDouble(static_cast<double>(min_value))});
      metrics.push_back(
          {prefix + "_max", FormatDouble(static_cast<double>(max_value))});
      metrics.push_back({prefix + "_checksum", FormatDouble(checksum)});
    }
  }

  return metrics;
}

void OnnxBackend::Unload() {
  FreeCudaBuffers();
  outputs_.clear();
  inputs_.clear();
  session_.reset();
  session_options_.reset();
  env_.reset();
}

BackendRunResult OnnxBackend::ValidateLoaded() const {
  if (session_ == nullptr) {
    return ErrorResult("ONNX backend is not loaded");
  }
  if (inputs_.empty()) {
    return ErrorResult("ONNX model has no supported inputs");
  }
  if (outputs_.empty()) {
    return ErrorResult("ONNX model has no supported outputs");
  }
  return OkResult();
}

BackendRunResult OnnxBackend::DiscoverModelIo() {
  Ort::AllocatorWithDefaultOptions allocator;
  const std::size_t input_count = session_->GetInputCount();
  const std::size_t output_count = session_->GetOutputCount();
  if (input_count == 0) {
    return ErrorResult("ONNX model has zero inputs");
  }
  if (output_count == 0) {
    return ErrorResult("ONNX model has zero outputs");
  }

  inputs_.clear();
  outputs_.clear();
  inputs_.reserve(input_count);
  outputs_.reserve(output_count);

  for (std::size_t i = 0; i < input_count; ++i) {
    BackendRunResult result = DiscoverInput(i, &allocator);
    if (!result.ok) {
      return result;
    }
  }
  for (std::size_t i = 0; i < output_count; ++i) {
    BackendRunResult result = DiscoverOutput(i, &allocator);
    if (!result.ok) {
      return result;
    }
  }
  return OkResult();
}

BackendRunResult OnnxBackend::DiscoverInput(
    std::size_t index, Ort::AllocatorWithDefaultOptions* allocator) {
  Ort::AllocatedStringPtr name =
      session_->GetInputNameAllocated(static_cast<int>(index), *allocator);
  Ort::TypeInfo type_info = session_->GetInputTypeInfo(index);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

  TensorBuffer tensor;
  tensor.name = name.get();
  tensor.element_type = tensor_info.GetElementType();
  tensor.shape = tensor_info.GetShape();

  BackendRunResult result = ResolveShape(&tensor);
  if (!result.ok) {
    return result;
  }
  if (tensor.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::ostringstream message;
    message << "unsupported input type for " << tensor.name << ": "
            << ElementTypeName(tensor.element_type);
    return ErrorResult(message.str());
  }
  result = ComputeElementCount(tensor.shape, &tensor.element_count);
  if (!result.ok) {
    return result;
  }
  result =
      FillInput(&tensor, options_.seed + static_cast<std::uint32_t>(index));
  if (!result.ok) {
    return result;
  }
  inputs_.push_back(std::move(tensor));
  return OkResult();
}

BackendRunResult OnnxBackend::DiscoverOutput(
    std::size_t index, Ort::AllocatorWithDefaultOptions* allocator) {
  Ort::AllocatedStringPtr name =
      session_->GetOutputNameAllocated(static_cast<int>(index), *allocator);
  Ort::TypeInfo type_info = session_->GetOutputTypeInfo(index);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

  TensorBuffer tensor;
  tensor.name = name.get();
  tensor.element_type = tensor_info.GetElementType();
  tensor.shape = tensor_info.GetShape();

  BackendRunResult result = ResolveShape(&tensor);
  if (!result.ok) {
    return result;
  }
  if (tensor.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::ostringstream message;
    message << "unsupported output type for " << tensor.name << ": "
            << ElementTypeName(tensor.element_type);
    return ErrorResult(message.str());
  }
  result = ComputeElementCount(tensor.shape, &tensor.element_count);
  if (!result.ok) {
    return result;
  }
  tensor.host_data.resize(tensor.element_count);
  outputs_.push_back(std::move(tensor));
  return OkResult();
}

BackendRunResult OnnxBackend::ResolveShape(TensorBuffer* tensor) const {
  bool has_dynamic_dim = false;
  for (const int64_t dim : tensor->shape) {
    if (dim <= 0) {
      has_dynamic_dim = true;
      break;
    }
  }
  if (!has_dynamic_dim) {
    return OkResult();
  }

  const OnnxShapeOverride* override_shape = FindShapeOverride(tensor->name);
  if (override_shape == nullptr) {
    std::ostringstream message;
    message << "dynamic shape for " << tensor->name << " requires --shape "
            << tensor->name << "=...";
    return ErrorResult(message.str());
  }
  if (override_shape->shape.size() != tensor->shape.size()) {
    std::ostringstream message;
    message << "shape override rank mismatch for " << tensor->name;
    return ErrorResult(message.str());
  }
  tensor->shape = override_shape->shape;
  return OkResult();
}

BackendRunResult OnnxBackend::FillInput(TensorBuffer* tensor,
                                        std::uint32_t seed) const {
  tensor->host_data.resize(tensor->element_count);
  switch (options_.input_mode) {
    case OnnxInputMode::kZero:
      for (float& value : tensor->host_data) {
        value = 0.0f;
      }
      return OkResult();
    case OnnxInputMode::kOne:
      for (float& value : tensor->host_data) {
        value = 1.0f;
      }
      return OkResult();
    case OnnxInputMode::kRamp:
      for (std::size_t i = 0; i < tensor->host_data.size(); ++i) {
        tensor->host_data[i] = static_cast<float>(i % 255) / 255.0f;
      }
      return OkResult();
    case OnnxInputMode::kRandom: {
      std::mt19937 generator(seed);
      std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
      for (float& value : tensor->host_data) {
        value = distribution(generator);
      }
      return OkResult();
    }
  }
  return ErrorResult("unknown ONNX input generation mode");
}

BackendRunResult OnnxBackend::AllocateCudaBuffers() {
  try {
    for (TensorBuffer& input : inputs_) {
      const std::size_t byte_count = input.host_data.size() * sizeof(float);
      MOSAICVRAM_CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void**>(&input.device_data), byte_count),
          "allocate ONNX CUDA input buffer");
      MOSAICVRAM_CUDA_CHECK(
          cudaMemcpy(input.device_data, input.host_data.data(), byte_count,
                     cudaMemcpyHostToDevice),
          "copy ONNX input to CUDA buffer");
    }

    for (TensorBuffer& output : outputs_) {
      const std::size_t byte_count = output.host_data.size() * sizeof(float);
      MOSAICVRAM_CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void**>(&output.device_data), byte_count),
          "allocate ONNX CUDA output buffer");
    }
    return OkResult();
  } catch (const std::exception& error) {
    FreeCudaBuffers();
    return ErrorResult(error.what());
  }
}

void OnnxBackend::FreeCudaBuffers() {
  for (TensorBuffer& output : outputs_) {
    if (output.device_data != nullptr) {
      cudaFree(output.device_data);
      output.device_data = nullptr;
    }
  }
  for (TensorBuffer& input : inputs_) {
    if (input.device_data != nullptr) {
      cudaFree(input.device_data);
      input.device_data = nullptr;
    }
  }
}

BackendRunResult OnnxBackend::RunCpuOnce() {
  try {
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_values;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    input_values.reserve(inputs_.size());
    input_names.reserve(inputs_.size());
    output_names.reserve(outputs_.size());

    for (TensorBuffer& input : inputs_) {
      input_values.push_back(Ort::Value::CreateTensor<float>(
          memory_info, input.host_data.data(), input.host_data.size(),
          input.shape.data(), input.shape.size()));
      input_names.push_back(input.name.c_str());
    }
    for (const TensorBuffer& output : outputs_) {
      output_names.push_back(output.name.c_str());
    }

    std::vector<Ort::Value> output_values = session_->Run(
        Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
        input_values.size(), output_names.data(), output_names.size());

    if (output_values.size() != outputs_.size()) {
      return ErrorResult("ONNX Runtime returned an unexpected output count");
    }
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      const float* output = output_values[i].GetTensorData<float>();
      outputs_[i].host_data.assign(output, output + outputs_[i].element_count);
    }
    return ValidateOutputs(outputs_);
  } catch (const std::exception& error) {
    return ErrorResult(error.what());
  }
}

BackendRunResult OnnxBackend::RunCudaOnce() {
  try {
    Ort::MemoryInfo cuda_memory_info("Cuda", OrtDeviceAllocator,
                                     options_.device_index, OrtMemTypeDefault);
    Ort::IoBinding binding(*session_);
    std::vector<Ort::Value> input_values;
    std::vector<Ort::Value> output_values;
    input_values.reserve(inputs_.size());
    output_values.reserve(outputs_.size());

    for (TensorBuffer& input : inputs_) {
      if (input.device_data == nullptr) {
        return ErrorResult("ONNX CUDA input buffer is not allocated");
      }
      input_values.push_back(Ort::Value::CreateTensor<float>(
          cuda_memory_info, input.device_data, input.element_count,
          input.shape.data(), input.shape.size()));
      binding.BindInput(input.name.c_str(), input_values.back());
    }

    for (TensorBuffer& output : outputs_) {
      if (output.device_data == nullptr) {
        return ErrorResult("ONNX CUDA output buffer is not allocated");
      }
      output_values.push_back(Ort::Value::CreateTensor<float>(
          cuda_memory_info, output.device_data, output.element_count,
          output.shape.data(), output.shape.size()));
      binding.BindOutput(output.name.c_str(), output_values.back());
    }

    session_->Run(Ort::RunOptions{nullptr}, binding);

    for (TensorBuffer& output : outputs_) {
      MOSAICVRAM_CUDA_CHECK(
          cudaMemcpy(output.host_data.data(), output.device_data,
                     output.host_data.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy ONNX output from CUDA buffer");
    }
    return ValidateOutputs(outputs_);
  } catch (const std::exception& error) {
    return ErrorResult(error.what());
  }
}

BackendRunResult OnnxBackend::ValidateOutputs(
    const std::vector<TensorBuffer>& outputs) const {
  for (const TensorBuffer& output : outputs) {
    if (output.host_data.empty()) {
      std::ostringstream message;
      message << "ONNX output is empty: " << output.name;
      return ErrorResult(message.str());
    }
    if (!IsFiniteVector(output.host_data)) {
      std::ostringstream message;
      message << "ONNX output contains NaN or Inf: " << output.name;
      return ErrorResult(message.str());
    }
  }
  return OkResult();
}

const OnnxShapeOverride* OnnxBackend::FindShapeOverride(
    const std::string& name) const {
  for (const OnnxShapeOverride& override_shape : options_.shape_overrides) {
    if (override_shape.name == name) {
      return &override_shape;
    }
  }
  return nullptr;
}

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_ONNX
