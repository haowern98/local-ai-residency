#include "residency/onnx_llm_residency_adapter.h"

#ifdef MOSAICVRAM_ENABLE_ONNX

#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "cuda/cuda_error.h"
#include "util/timer.h"

namespace mosaicvram {
namespace {

constexpr std::size_t kFloat16Bytes = 2;
constexpr std::size_t kFloat32Bytes = 4;

std::wstring ToWideString(const std::string& text) {
  return std::wstring(text.begin(), text.end());
}

std::size_t ElementCount(const std::vector<int64_t>& shape) {
  std::size_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0) {
      throw std::runtime_error("cannot count unresolved ONNX tensor shape");
    }
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

bool StartsWith(std::string_view text, std::string_view prefix) {
  return text.substr(0, prefix.size()) == prefix;
}

bool EndsWith(std::string_view text, std::string_view suffix) {
  if (text.size() < suffix.size()) {
    return false;
  }
  return text.substr(text.size() - suffix.size()) == suffix;
}

bool ParseKvName(const std::string& name, const std::string& prefix, int* layer,
                 bool* is_key) {
  if (!StartsWith(name, prefix)) {
    return false;
  }
  std::string_view rest(name.data() + prefix.size(),
                        name.size() - prefix.size());
  const std::size_t dot_pos = rest.find('.');
  if (dot_pos == std::string_view::npos || dot_pos == 0) {
    return false;
  }
  int parsed_layer = 0;
  const std::string_view layer_text = rest.substr(0, dot_pos);
  const char* begin = layer_text.data();
  const char* end = begin + layer_text.size();
  const auto result = std::from_chars(begin, end, parsed_layer);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }

  const std::string_view kind = rest.substr(dot_pos + 1);
  if (kind == "key") {
    *layer = parsed_layer;
    *is_key = true;
    return true;
  }
  if (kind == "value") {
    *layer = parsed_layer;
    *is_key = false;
    return true;
  }
  return false;
}

float HalfToFloat(std::uint16_t half) {
  const std::uint32_t sign = (half >> 15) & 0x1;
  const std::uint32_t exponent = (half >> 10) & 0x1f;
  const std::uint32_t mantissa = half & 0x3ff;

  if (exponent == 0) {
    if (mantissa == 0) {
      return sign == 0 ? 0.0f : -0.0f;
    }
    const float value = std::ldexp(static_cast<float>(mantissa), -24);
    return sign == 0 ? value : -value;
  }
  if (exponent == 31) {
    if (mantissa == 0) {
      return sign == 0 ? std::numeric_limits<float>::infinity()
                       : -std::numeric_limits<float>::infinity();
    }
    return std::numeric_limits<float>::quiet_NaN();
  }

  const float value = std::ldexp(static_cast<float>(mantissa) / 1024.0f + 1.0f,
                                 static_cast<int>(exponent) - 15);
  return sign == 0 ? value : -value;
}

const char* TensorElementTypeName(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    default:
      return "unsupported";
  }
}

std::size_t TensorElementBytes(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return kFloat32Bytes;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return kFloat16Bytes;
    default:
      throw std::runtime_error("unsupported ONNX LLM tensor element type");
  }
}

void CountLogitValue(float value, OnnxLlmResidencyReport* report) {
  if (std::isnan(value)) {
    ++report->logits_nan_count;
    return;
  }
  if (std::isinf(value)) {
    if (value > 0.0f) {
      ++report->logits_pos_inf_count;
    } else {
      ++report->logits_neg_inf_count;
    }
    return;
  }
  ++report->logits_finite_count;
}

bool ContainsSubsequence(const std::vector<int64_t>& values,
                         const std::vector<int64_t>& expected) {
  if (expected.empty() || values.size() < expected.size()) {
    return false;
  }
  return std::search(values.begin(), values.end(), expected.begin(),
                     expected.end()) != values.end();
}

std::string DecodeFailureMessage(const OnnxLlmResidencyReport& report) {
  std::ostringstream message;
  message << "ONNX LLM decode produced no valid next token"
          << "; logits_dtype=" << report.logits_dtype
          << "; vocab_size=" << report.logits_vocab_size
          << "; finite=" << report.logits_finite_count
          << "; nan=" << report.logits_nan_count
          << "; pos_inf=" << report.logits_pos_inf_count
          << "; neg_inf=" << report.logits_neg_inf_count
          << "; position_ids_present="
          << (report.position_ids_present ? "yes" : "no");
  return message.str();
}

}  // namespace

OnnxLlmResidencyAdapter::CudaBuffer::~CudaBuffer() { Free(); }

OnnxLlmResidencyAdapter::CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept {
  data = other.data;
  bytes = other.bytes;
  other.data = nullptr;
  other.bytes = 0;
}

OnnxLlmResidencyAdapter::CudaBuffer&
OnnxLlmResidencyAdapter::CudaBuffer::operator=(CudaBuffer&& other) noexcept {
  if (this != &other) {
    Free();
    data = other.data;
    bytes = other.bytes;
    other.data = nullptr;
    other.bytes = 0;
  }
  return *this;
}

void OnnxLlmResidencyAdapter::CudaBuffer::Allocate(std::size_t byte_count) {
  Free();
  bytes = byte_count;
  const std::size_t allocation_bytes = byte_count == 0 ? 1 : byte_count;
  MOSAICVRAM_CUDA_CHECK(cudaMalloc(&data, allocation_bytes),
                        "allocate ONNX LLM CUDA buffer");
}

void OnnxLlmResidencyAdapter::CudaBuffer::Free() {
  if (data != nullptr) {
    cudaFree(data);
    data = nullptr;
    bytes = 0;
  }
}

OnnxLlmResidencyAdapter::OnnxLlmResidencyAdapter(
    OnnxLlmResidencyOptions options)
    : options_(std::move(options)) {
  snapshot_.session_id = options_.session_id;
  snapshot_.backend_name = "onnx-llm";
}

OnnxLlmResidencyAdapter::~OnnxLlmResidencyAdapter() = default;

void OnnxLlmResidencyAdapter::Load() {
  if (options_.prefill_chunk_tokens <= 0) {
    throw std::runtime_error("ONNX LLM prefill chunk size must be positive");
  }
  MOSAICVRAM_CUDA_CHECK(cudaSetDevice(options_.device_index),
                        "set ONNX LLM CUDA device");

  double load_ms = 0.0;
  CreateSession(&load_ms);
  report_.initial_session_load_ms = load_ms;
  DiscoverModelIo();
  AllocateInitialCache();
  residency_state_ = ResidencyState::kResident;
}

void OnnxLlmResidencyAdapter::PrefillPrompt() {
  Timer timer;
  ValidateReadyForDecode();
  if (options_.prompt_tokens.empty()) {
    throw std::runtime_error("ONNX LLM prompt tokens are required");
  }
  DecodeResult result;
  const std::size_t chunk_size =
      static_cast<std::size_t>(options_.prefill_chunk_tokens);
  std::size_t token_offset = 0;
  std::size_t chunk_count = 0;
  while (token_offset < options_.prompt_tokens.size()) {
    const std::size_t remaining = options_.prompt_tokens.size() - token_offset;
    const std::size_t current_chunk_size = std::min(chunk_size, remaining);
    const bool final_chunk =
        token_offset + current_chunk_size == options_.prompt_tokens.size();
    const auto begin = options_.prompt_tokens.begin() +
                       static_cast<std::ptrdiff_t>(token_offset);
    const auto end = begin + static_cast<std::ptrdiff_t>(current_chunk_size);
    const std::vector<int64_t> chunk_tokens(begin, end);
    result = RunDecodeStep(chunk_tokens, cache_length_, final_chunk);
    token_offset += current_chunk_size;
    ++chunk_count;
  }
  report_.first_token = result.next_token;
  report_.prompt_tokens = options_.prompt_tokens.size();
  report_.prefill_chunks = chunk_count;
  report_.prefill_chunk_tokens = options_.prefill_chunk_tokens;
  report_.prefill_ms = timer.ElapsedMs();
}

void OnnxLlmResidencyAdapter::CaptureBaselineNextToken() {
  ValidateReadyForDecode();
  const DecodeResult result =
      RunDecodeStep({report_.first_token}, cache_length_, true);
  report_.baseline_next_token = result.next_token;
  report_.baseline_logits_checksum = result.logits_checksum;
}

BackendStateSnapshot& OnnxLlmResidencyAdapter::SaveState() {
  Timer timer;
  ValidateReadyForDecode();
  const std::size_t bytes = CacheBytes();
  if (bytes == 0) {
    throw std::runtime_error("ONNX LLM context is empty; nothing to save");
  }
  snapshot_.full_state.Allocate(bytes);
  std::uint8_t* dst = snapshot_.full_state.data();
  for (const KvTensor& tensor : kv_tensors_) {
    MOSAICVRAM_CUDA_CHECK(
        cudaMemcpy(dst, tensor.cache_buffer.data, tensor.cache_buffer.bytes,
                   cudaMemcpyDeviceToHost),
        "copy ONNX LLM KV cache to pinned host memory");
    dst += tensor.cache_buffer.bytes;
  }
  snapshot_.full_state_bytes = bytes;
  snapshot_.source_residency = residency_state_;
  snapshot_cache_length_ = cache_length_;
  report_.kv_state_bytes = bytes;
  report_.save_state_ms = timer.ElapsedMs();
  return snapshot_;
}

void OnnxLlmResidencyAdapter::EvictContext() {
  Timer timer;
  FreeCache();
  residency_state_ = session_ == nullptr ? ResidencyState::kModelEvicted
                                         : ResidencyState::kContextEvicted;
  report_.evict_context_ms = timer.ElapsedMs();
}

void OnnxLlmResidencyAdapter::EvictModel() {
  Timer timer;
  FreeCache();
  session_.reset();
  session_options_.reset();
  env_.reset();
  residency_state_ = ResidencyState::kModelEvicted;
  report_.evict_model_ms = timer.ElapsedMs();
}

void OnnxLlmResidencyAdapter::ReloadModel() {
  double load_ms = 0.0;
  CreateSession(&load_ms);
  report_.session_reload_ms = load_ms;
  DiscoverModelIo();
  residency_state_ = ResidencyState::kContextEvicted;
}

void OnnxLlmResidencyAdapter::RestoreState() {
  Timer timer;
  if (snapshot_.full_state.empty()) {
    throw std::runtime_error("ONNX LLM KV snapshot is empty");
  }
  cache_length_ = snapshot_cache_length_;
  std::uint8_t* src = snapshot_.full_state.data();
  std::size_t copied = 0;
  for (KvTensor& tensor : kv_tensors_) {
    std::vector<int64_t> shape = tensor.base_shape;
    shape[0] = 1;
    shape[2] = cache_length_;
    tensor.cache_shape = shape;
    tensor.cache_buffer.Allocate(ElementCount(shape) *
                                 TensorElementBytes(tensor.element_type));
    MOSAICVRAM_CUDA_CHECK(
        cudaMemcpy(tensor.cache_buffer.data, src, tensor.cache_buffer.bytes,
                   cudaMemcpyHostToDevice),
        "restore ONNX LLM KV cache from pinned host memory");
    src += tensor.cache_buffer.bytes;
    copied += tensor.cache_buffer.bytes;
  }
  report_.restored_kv_state_bytes = copied;
  residency_state_ = ResidencyState::kResident;
  report_.restore_state_ms = timer.ElapsedMs();
}

bool OnnxLlmResidencyAdapter::ResumeCheck() {
  Timer timer;
  ValidateReadyForDecode();
  const DecodeResult result =
      RunDecodeStep({report_.first_token}, cache_length_, true);
  report_.restored_next_token = result.next_token;
  report_.restored_logits_checksum = result.logits_checksum;
  report_.resume_match = report_.baseline_next_token >= 0 &&
                         result.next_token >= 0 &&
                         report_.baseline_next_token == result.next_token;
  report_.resume_check_ms = timer.ElapsedMs();
  return report_.resume_match;
}

void OnnxLlmResidencyAdapter::RestoreAndGenerateContinuation(
    const std::vector<int64_t>& tokens, int max_tokens,
    const std::vector<int64_t>& expected_tokens) {
  Timer timer;
  if (tokens.empty()) {
    throw std::runtime_error("ONNX LLM continuation tokens are required");
  }
  if (max_tokens <= 0) {
    throw std::runtime_error("max_tokens must be positive");
  }
  if (session_ == nullptr) {
    ReloadModel();
  }
  RestoreState();

  DecodeResult next = RunDecodeStep(tokens, cache_length_, true);
  report_.generated_token_ids.clear();
  report_.generated_token_ids.reserve(static_cast<std::size_t>(max_tokens));
  for (int i = 0; i < max_tokens; ++i) {
    report_.generated_token_ids.push_back(next.next_token);
    next = RunDecodeStep({next.next_token}, cache_length_, true);
  }

  report_.generated_tokens = report_.generated_token_ids.size();
  report_.generated_contains_expected =
      ContainsSubsequence(report_.generated_token_ids, expected_tokens);
  report_.restore_generate_ms = timer.ElapsedMs();
}

void OnnxLlmResidencyAdapter::CreateSession(double* elapsed_ms) {
  Timer timer;
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                    "mosaicvram-onnx-llm");
  session_options_ = std::make_unique<Ort::SessionOptions>();
  session_options_->SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = options_.device_index;
  session_options_->AppendExecutionProvider_CUDA(cuda_options);

  const std::wstring model_path = ToWideString(options_.model_path);
  session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(),
                                            *session_options_);
  *elapsed_ms = timer.ElapsedMs();
}

void OnnxLlmResidencyAdapter::DiscoverModelIo() {
  input_ids_name_.clear();
  attention_mask_name_.clear();
  position_ids_name_.clear();
  logits_name_.clear();
  kv_tensors_.clear();
  logits_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  vocab_size_ = 0;
  report_.past_input_count = 0;
  report_.present_output_count = 0;
  report_.logits_output_count = 0;
  report_.logits_dtype.clear();
  report_.logits_vocab_size = 0;
  report_.logits_finite_count = 0;
  report_.logits_nan_count = 0;
  report_.logits_pos_inf_count = 0;
  report_.logits_neg_inf_count = 0;
  report_.position_ids_present = false;
  report_.decode_valid = false;
  report_.cache_surface_found = false;

  struct Pair {
    std::string past_name;
    std::string present_name;
    ONNXTensorElementDataType past_element_type =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType present_element_type =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<int64_t> shape;
  };

  std::map<std::tuple<int, bool>, Pair> kv_pairs;
  Ort::AllocatorWithDefaultOptions allocator;
  const std::size_t input_count = session_->GetInputCount();
  const std::size_t output_count = session_->GetOutputCount();

  for (std::size_t i = 0; i < input_count; ++i) {
    Ort::AllocatedStringPtr name =
        session_->GetInputNameAllocated(i, allocator);
    const std::string input_name = name.get();
    Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
    Ort::ConstTensorTypeAndShapeInfo tensor_info =
        type_info.GetTensorTypeAndShapeInfo();

    if (input_name == "input_ids") {
      input_ids_name_ = input_name;
      continue;
    }
    if (input_name == "attention_mask") {
      attention_mask_name_ = input_name;
      continue;
    }
    if (input_name == "position_ids") {
      position_ids_name_ = input_name;
      report_.position_ids_present = true;
      continue;
    }

    int layer = 0;
    bool is_key = false;
    if (ParseKvName(input_name, "past_key_values.", &layer, &is_key)) {
      const ONNXTensorElementDataType element_type =
          tensor_info.GetElementType();
      if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
          element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::ostringstream message;
        message << "ONNX LLM KV input " << input_name << " has unsupported "
                << TensorElementTypeName(element_type) << " dtype";
        throw std::runtime_error(message.str());
      }
      Pair& pair = kv_pairs[std::make_tuple(layer, is_key)];
      pair.past_name = input_name;
      pair.past_element_type = element_type;
      pair.shape = tensor_info.GetShape();
      ++report_.past_input_count;
      continue;
    }

    std::ostringstream message;
    message << "ONNX LLM required input is unsupported: " << input_name;
    throw std::runtime_error(message.str());
  }

  for (std::size_t i = 0; i < output_count; ++i) {
    Ort::AllocatedStringPtr name =
        session_->GetOutputNameAllocated(i, allocator);
    const std::string output_name = name.get();
    Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
    Ort::ConstTensorTypeAndShapeInfo tensor_info =
        type_info.GetTensorTypeAndShapeInfo();
    if (output_name == "logits") {
      logits_name_ = output_name;
      const std::vector<int64_t> shape = tensor_info.GetShape();
      if (shape.size() != 3 || shape[2] <= 0) {
        throw std::runtime_error("ONNX LLM logits shape is unsupported");
      }
      vocab_size_ = shape[2];
      report_.logits_vocab_size = static_cast<std::size_t>(vocab_size_);
      logits_element_type_ = tensor_info.GetElementType();
      if (logits_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
          logits_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::ostringstream message;
        message << "ONNX LLM logits dtype is unsupported: "
                << TensorElementTypeName(logits_element_type_);
        throw std::runtime_error(message.str());
      }
      report_.logits_dtype = TensorElementTypeName(logits_element_type_);
      ++report_.logits_output_count;
      continue;
    }

    int layer = 0;
    bool is_key = false;
    if (ParseKvName(output_name, "present.", &layer, &is_key)) {
      const ONNXTensorElementDataType element_type =
          tensor_info.GetElementType();
      if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
          element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::ostringstream message;
        message << "ONNX LLM KV output " << output_name << " has unsupported "
                << TensorElementTypeName(element_type) << " dtype";
        throw std::runtime_error(message.str());
      }
      Pair& pair = kv_pairs[std::make_tuple(layer, is_key)];
      pair.present_name = output_name;
      pair.present_element_type = element_type;
      ++report_.present_output_count;
    }
  }

  for (const auto& [unused_key, pair] : kv_pairs) {
    (void)unused_key;
    if (pair.past_name.empty() || pair.present_name.empty()) {
      throw std::runtime_error("ONNX LLM KV past/present pair is incomplete");
    }
    if (pair.past_element_type != pair.present_element_type) {
      throw std::runtime_error("ONNX LLM KV past/present dtypes differ");
    }
    if (pair.shape.size() != 4 || pair.shape[1] <= 0 || pair.shape[3] <= 0) {
      throw std::runtime_error("ONNX LLM KV shape is unsupported");
    }
    KvTensor tensor;
    tensor.past_name = pair.past_name;
    tensor.present_name = pair.present_name;
    tensor.element_type = pair.past_element_type;
    tensor.base_shape = pair.shape;
    kv_tensors_.push_back(std::move(tensor));
  }

  report_.cache_surface_found =
      !input_ids_name_.empty() && !attention_mask_name_.empty() &&
      !logits_name_.empty() && report_.past_input_count > 0 &&
      report_.present_output_count == report_.past_input_count;
  if (!report_.cache_surface_found) {
    throw std::runtime_error("ONNX LLM cache surface was not found");
  }
}

void OnnxLlmResidencyAdapter::AllocateInitialCache() {
  for (KvTensor& tensor : kv_tensors_) {
    std::vector<int64_t> shape = tensor.base_shape;
    shape[0] = 1;
    shape[2] = cache_length_;
    tensor.cache_shape = shape;
    tensor.cache_buffer.Allocate(ElementCount(shape) *
                                 TensorElementBytes(tensor.element_type));
  }
}

OnnxLlmResidencyAdapter::DecodeResult OnnxLlmResidencyAdapter::RunDecodeStep(
    const std::vector<int64_t>& input_tokens, int64_t past_length,
    bool read_logits) {
  if (input_tokens.empty()) {
    throw std::runtime_error("ONNX LLM decode requires tokens");
  }

  const int64_t input_length = static_cast<int64_t>(input_tokens.size());
  const int64_t total_length = past_length + input_length;
  Ort::MemoryInfo cpu_memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::MemoryInfo cuda_memory_info("Cuda", OrtDeviceAllocator,
                                   options_.device_index, OrtMemTypeDefault);
  Ort::IoBinding binding(*session_);
  std::vector<Ort::Value> input_values;
  std::vector<Ort::Value> output_values;
  input_values.reserve(3 + kv_tensors_.size());
  output_values.reserve(1 + kv_tensors_.size());

  const std::vector<int64_t> input_shape = {1, input_length};
  const std::vector<int64_t> mask_shape = {1, total_length};
  std::vector<int64_t> mask(static_cast<std::size_t>(total_length), 1);

  input_values.push_back(Ort::Value::CreateTensor<int64_t>(
      cpu_memory_info, const_cast<int64_t*>(input_tokens.data()),
      input_tokens.size(), input_shape.data(), input_shape.size()));
  binding.BindInput(input_ids_name_.c_str(), input_values.back());

  input_values.push_back(Ort::Value::CreateTensor<int64_t>(
      cpu_memory_info, mask.data(), mask.size(), mask_shape.data(),
      mask_shape.size()));
  binding.BindInput(attention_mask_name_.c_str(), input_values.back());

  if (!position_ids_name_.empty()) {
    std::vector<int64_t> positions(static_cast<std::size_t>(input_length));
    for (int64_t i = 0; i < input_length; ++i) {
      positions[static_cast<std::size_t>(i)] = past_length + i;
    }
    input_values.push_back(Ort::Value::CreateTensor<int64_t>(
        cpu_memory_info, positions.data(), positions.size(), input_shape.data(),
        input_shape.size()));
    binding.BindInput(position_ids_name_.c_str(), input_values.back());
  }

  for (KvTensor& tensor : kv_tensors_) {
    input_values.push_back(Ort::Value::CreateTensor(
        cuda_memory_info, tensor.cache_buffer.data, tensor.cache_buffer.bytes,
        tensor.cache_shape.data(), tensor.cache_shape.size(),
        tensor.element_type));
    binding.BindInput(tensor.past_name.c_str(), input_values.back());
  }

  CudaBuffer logits_buffer;
  const std::vector<int64_t> logits_shape = {1, input_length, vocab_size_};
  logits_buffer.Allocate(ElementCount(logits_shape) *
                         TensorElementBytes(logits_element_type_));
  output_values.push_back(Ort::Value::CreateTensor(
      cuda_memory_info, logits_buffer.data, logits_buffer.bytes,
      logits_shape.data(), logits_shape.size(), logits_element_type_));
  binding.BindOutput(logits_name_.c_str(), output_values.back());

  std::vector<KvTensor> present_tensors;
  present_tensors.reserve(kv_tensors_.size());
  for (const KvTensor& tensor : kv_tensors_) {
    KvTensor present;
    present.past_name = tensor.past_name;
    present.present_name = tensor.present_name;
    present.element_type = tensor.element_type;
    present.base_shape = tensor.base_shape;
    present.cache_shape = tensor.base_shape;
    present.cache_shape[0] = 1;
    present.cache_shape[2] = total_length;
    present.cache_buffer.Allocate(ElementCount(present.cache_shape) *
                                  TensorElementBytes(present.element_type));
    output_values.push_back(Ort::Value::CreateTensor(
        cuda_memory_info, present.cache_buffer.data, present.cache_buffer.bytes,
        present.cache_shape.data(), present.cache_shape.size(),
        present.element_type));
    binding.BindOutput(present.present_name.c_str(), output_values.back());
    present_tensors.push_back(std::move(present));
  }

  session_->Run(Ort::RunOptions{nullptr}, binding);

  const std::size_t logits_values =
      static_cast<std::size_t>(input_length * vocab_size_);
  const std::size_t last_token_offset =
      static_cast<std::size_t>((input_length - 1) * vocab_size_);
  float best_logit = -std::numeric_limits<float>::infinity();
  int64_t best_token = -1;
  double checksum = 0.0;
  report_.logits_finite_count = 0;
  report_.logits_nan_count = 0;
  report_.logits_pos_inf_count = 0;
  report_.logits_neg_inf_count = 0;
  report_.decode_valid = false;

  if (!read_logits) {
    ReplaceCache(&present_tensors, total_length);
    return DecodeResult{-1, 0.0};
  }

  if (logits_element_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    std::vector<std::uint16_t> host_logits(logits_values);
    MOSAICVRAM_CUDA_CHECK(
        cudaMemcpy(host_logits.data(), logits_buffer.data, logits_buffer.bytes,
                   cudaMemcpyDeviceToHost),
        "copy ONNX LLM float16 logits to host");
    for (int64_t i = 0; i < vocab_size_; ++i) {
      const float value = HalfToFloat(
          host_logits[last_token_offset + static_cast<std::size_t>(i)]);
      CountLogitValue(value, &report_);
      if (!std::isfinite(value)) {
        continue;
      }
      checksum += static_cast<double>(value);
      if (value > best_logit) {
        best_logit = value;
        best_token = i;
      }
    }
  } else if (logits_element_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::vector<float> host_logits(logits_values);
    MOSAICVRAM_CUDA_CHECK(
        cudaMemcpy(host_logits.data(), logits_buffer.data, logits_buffer.bytes,
                   cudaMemcpyDeviceToHost),
        "copy ONNX LLM float32 logits to host");
    for (int64_t i = 0; i < vocab_size_; ++i) {
      const float value =
          host_logits[last_token_offset + static_cast<std::size_t>(i)];
      CountLogitValue(value, &report_);
      if (!std::isfinite(value)) {
        continue;
      }
      checksum += static_cast<double>(value);
      if (value > best_logit) {
        best_logit = value;
        best_token = i;
      }
    }
  } else {
    throw std::runtime_error("ONNX LLM logits dtype is unsupported");
  }

  if (best_token < 0) {
    throw std::runtime_error(DecodeFailureMessage(report_));
  }
  report_.decode_valid = true;

  ReplaceCache(&present_tensors, total_length);
  return DecodeResult{best_token, checksum};
}

void OnnxLlmResidencyAdapter::ReplaceCache(
    std::vector<KvTensor>* output_tensors, int64_t cache_length) {
  kv_tensors_ = std::move(*output_tensors);
  cache_length_ = cache_length;
}

void OnnxLlmResidencyAdapter::FreeCache() {
  for (KvTensor& tensor : kv_tensors_) {
    tensor.cache_buffer.Free();
  }
}

void OnnxLlmResidencyAdapter::ValidateReadyForDecode() const {
  if (session_ == nullptr || kv_tensors_.empty() || vocab_size_ <= 0) {
    throw std::runtime_error("ONNX LLM adapter is not ready");
  }
}

std::size_t OnnxLlmResidencyAdapter::CacheBytes() const {
  std::size_t bytes = 0;
  for (const KvTensor& tensor : kv_tensors_) {
    bytes += tensor.cache_buffer.bytes;
  }
  return bytes;
}

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_ONNX
