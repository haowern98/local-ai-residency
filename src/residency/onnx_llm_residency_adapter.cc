#include "residency/onnx_llm_residency_adapter.h"

#ifdef MOSAICVRAM_ENABLE_ONNX

#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "cuda/cuda_error.h"
#include "util/timer.h"

#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON
#include "tokenizer/tokenizers_cpp_adapter.h"
#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON

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

bool ParsePrefixedStateName(const std::string& name, const std::string& prefix,
                            std::string* suffix) {
  if (!StartsWith(name, prefix) || name.size() == prefix.size()) {
    return false;
  }
  *suffix = name.substr(prefix.size());
  return true;
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
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    default:
      return "unsupported";
  }
}

bool IsFloatTensor(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

bool IsIntTensor(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

std::size_t TensorElementBytes(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return kFloat32Bytes;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return kFloat16Bytes;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    default:
      throw std::runtime_error("unsupported ONNX LLM tensor element type");
  }
}

int64_t KnownHiddenSize(const std::vector<int64_t>& shape) {
  if (shape.size() != 3 || shape[2] <= 0) {
    return 0;
  }
  return shape[2];
}

bool ShapeCompatibleHiddenSize(const std::vector<int64_t>& shape,
                               int64_t expected_hidden_size) {
  if (expected_hidden_size <= 0 || shape.size() != 3) {
    return false;
  }
  return shape[2] <= 0 || shape[2] == expected_hidden_size;
}

std::vector<int64_t> StaticCacheShape(std::vector<int64_t> shape) {
  if (shape.empty()) {
    throw std::runtime_error("ONNX LLM recurrent state shape is unsupported");
  }
  for (int64_t& dim : shape) {
    if (dim < 0) {
      dim = 1;
    }
  }
  return shape;
}

bool FilenameLooksLikeEmbedding(const std::filesystem::path& path) {
  std::string filename = path.filename().string();
  std::transform(
      filename.begin(), filename.end(), filename.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return filename.find("embed") != std::string::npos ||
         filename.find("token") != std::string::npos;
}

std::vector<std::filesystem::path> CandidateOnnxFiles(
    const std::filesystem::path& decoder_model_path) {
  const std::filesystem::path directory = decoder_model_path.parent_path();
  const std::filesystem::path decoder_filename = decoder_model_path.filename();
  std::vector<std::filesystem::path> candidates;
  if (!std::filesystem::is_directory(directory)) {
    return candidates;
  }
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::filesystem::path path = entry.path();
    if (path.extension() != ".onnx" || path.filename() == decoder_filename) {
      continue;
    }
    candidates.push_back(path);
  }
  std::sort(
      candidates.begin(), candidates.end(),
      [](const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
        const bool lhs_embedding = FilenameLooksLikeEmbedding(lhs);
        const bool rhs_embedding = FilenameLooksLikeEmbedding(rhs);
        if (lhs_embedding != rhs_embedding) {
          return lhs_embedding;
        }
        return lhs.filename().string() < rhs.filename().string();
      });
  return candidates;
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

bool EndsWithTokens(const std::vector<int64_t>& values,
                    const std::vector<int64_t>& suffix) {
  if (suffix.empty() || values.size() < suffix.size()) {
    return false;
  }
  return std::equal(suffix.rbegin(), suffix.rend(), values.rbegin());
}

bool RemoveStopSuffix(const std::vector<std::vector<int64_t>>& stop_sequences,
                      std::vector<int64_t>* values) {
  for (const std::vector<int64_t>& sequence : stop_sequences) {
    if (EndsWithTokens(*values, sequence)) {
      values->resize(values->size() - sequence.size());
      return true;
    }
  }
  return false;
}

bool HasStopSuffix(const std::vector<std::vector<int64_t>>& stop_sequences,
                   const std::vector<int64_t>& values) {
  for (const std::vector<int64_t>& sequence : stop_sequences) {
    if (EndsWithTokens(values, sequence)) {
      return true;
    }
  }
  return false;
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

int64_t GreedyToken(std::span<const float> logits) {
  int64_t best_token = -1;
  float best_logit = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const float value = logits[i];
    if (!std::isfinite(value)) {
      continue;
    }
    if (best_token < 0 || value > best_logit) {
      best_token = static_cast<int64_t>(i);
      best_logit = value;
    }
  }
  return best_token;
}

bool IsBlank(std::string_view text) {
  for (const char c : text) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  return true;
}

void AppendMissingStrings(const std::vector<std::string>& values,
                          std::vector<std::string>* target) {
  for (const std::string& value : values) {
    if (std::find(target->begin(), target->end(), value) == target->end()) {
      target->push_back(value);
    }
  }
}

std::size_t FindFirstStopString(std::string_view text,
                                const std::vector<std::string>& stop_strings) {
  std::size_t first = std::string_view::npos;
  for (const std::string& stop : stop_strings) {
    const std::size_t found = text.find(stop);
    if (found != std::string_view::npos &&
        (first == std::string_view::npos || found < first)) {
      first = found;
    }
  }
  return first;
}

std::string ApplyStopStrings(std::string text,
                             const std::vector<std::string>& stop_strings) {
  const std::size_t stop = FindFirstStopString(text, stop_strings);
  if (stop != std::string_view::npos) {
    text.resize(stop);
  }
  return text;
}

#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON
void AppendUniqueTokenSequence(std::vector<int64_t> sequence,
                               std::vector<std::vector<int64_t>>* sequences) {
  if (sequence.empty()) {
    return;
  }
  if (std::find(sequences->begin(), sequences->end(), sequence) ==
      sequences->end()) {
    sequences->push_back(std::move(sequence));
  }
}

std::vector<std::vector<int64_t>> TokenizeStopStrings(
    const std::string& tokenizer_path,
    const std::vector<std::string>& stop_strings) {
  std::vector<std::vector<int64_t>> sequences;
  for (const std::string& stop : stop_strings) {
    const std::optional<int64_t> exact_token_id =
        TokenIdForTokenizerString(tokenizer_path, stop);
    if (exact_token_id.has_value()) {
      AppendUniqueTokenSequence({*exact_token_id}, &sequences);
      if (stop.starts_with("<")) {
        continue;
      }
    }

    std::vector<int64_t> tokens =
        TokenizeWithTokenizerJson(tokenizer_path, stop);
    if (!tokens.empty()) {
      if (stop.starts_with("<") && tokens.size() > 1) {
        AppendUniqueTokenSequence({tokens.back()}, &sequences);
      }
      AppendUniqueTokenSequence(std::move(tokens), &sequences);
    }
  }
  return sequences;
}
#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON

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
  cache_length_ = 0;
  chat_messages_.clear();
  chat_cache_tokens_.clear();
  AllocateInitialCache();
  residency_state_ = ResidencyState::kResident;
}

void OnnxLlmResidencyAdapter::ResetConversation() {
  chat_messages_.clear();
  chat_cache_tokens_.clear();
  FreeCache();
  if (session_ == nullptr) {
    residency_state_ = ResidencyState::kModelEvicted;
    return;
  }
  if (kv_tensors_.empty()) {
    DiscoverModelIo();
  }
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
  for (const RecurrentTensor& tensor : recurrent_tensors_) {
    MOSAICVRAM_CUDA_CHECK(
        cudaMemcpy(dst, tensor.buffer.data, tensor.buffer.bytes,
                   cudaMemcpyDeviceToHost),
        "copy ONNX LLM recurrent cache to pinned host memory");
    dst += tensor.buffer.bytes;
  }
  snapshot_.full_state_bytes = bytes;
  snapshot_.source_residency = residency_state_;
  snapshot_cache_length_ = cache_length_;
  snapshot_chat_messages_ = chat_messages_;
  snapshot_chat_cache_tokens_ = chat_cache_tokens_;
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
  embedding_session_.reset();
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
  for (RecurrentTensor& tensor : recurrent_tensors_) {
    tensor.buffer.Allocate(ElementCount(tensor.shape) *
                           TensorElementBytes(tensor.element_type));
    MOSAICVRAM_CUDA_CHECK(
        cudaMemcpy(tensor.buffer.data, src, tensor.buffer.bytes,
                   cudaMemcpyHostToDevice),
        "restore ONNX LLM recurrent cache from pinned host memory");
    src += tensor.buffer.bytes;
    copied += tensor.buffer.bytes;
  }
  report_.restored_kv_state_bytes = copied;
  chat_messages_ = snapshot_chat_messages_;
  chat_cache_tokens_ = snapshot_chat_cache_tokens_;
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

std::string OnnxLlmResidencyAdapter::GenerateChatReply(
    const std::string& tokenizer_path, const std::string& user_text,
    int max_tokens, const std::vector<std::string>& stop_strings) {
#ifndef MOSAICVRAM_ENABLE_TOKENIZER_JSON
  (void)tokenizer_path;
  (void)user_text;
  (void)max_tokens;
  (void)stop_strings;
  throw std::runtime_error(
      "ONNX chat requires tokenizer.json support in this build");
#else
  if (max_tokens <= 0) {
    throw std::runtime_error("max_tokens must be positive");
  }
  if (IsBlank(user_text)) {
    throw std::runtime_error("chat message is empty");
  }
  ValidateReadyForDecode();

  if (cache_length_ != static_cast<int64_t>(chat_cache_tokens_.size())) {
    throw std::runtime_error("ONNX chat cache metadata is out of sync");
  }

  const TokenizerChatPrompt chat_prompt = ApplyTokenizerChatTurnTemplate(
      tokenizer_path, user_text, chat_messages_.empty());
  const std::vector<int64_t> prompt_delta =
      TokenizeWithTokenizerJson(tokenizer_path, chat_prompt.text);

  std::vector<std::string> combined_stop_strings = stop_strings;
  AppendMissingStrings(chat_prompt.stop_strings, &combined_stop_strings);
  const std::vector<std::vector<int64_t>> stop_token_sequences =
      TokenizeStopStrings(tokenizer_path, combined_stop_strings);

  if (prompt_delta.empty()) {
    throw std::runtime_error("ONNX chat prompt produced no new tokens");
  }

  DecodeResult next = RunDecodeStep(prompt_delta, cache_length_, true);
  chat_cache_tokens_.insert(chat_cache_tokens_.end(), prompt_delta.begin(),
                            prompt_delta.end());

  LogitsSampler sampler(options_.sampling);
  std::vector<int64_t> recent_tokens = chat_cache_tokens_;
  std::vector<int64_t> generated_tokens;
  generated_tokens.reserve(static_cast<std::size_t>(max_tokens));

  for (int i = 0; i < max_tokens; ++i) {
    const int64_t sampled_token = sampler.Sample(next.logits, recent_tokens);
    std::vector<int64_t> candidate_tokens = generated_tokens;
    candidate_tokens.push_back(sampled_token);

    if (HasStopSuffix(stop_token_sequences, candidate_tokens)) {
      break;
    }

    generated_tokens.push_back(sampled_token);
    recent_tokens.push_back(sampled_token);
    chat_cache_tokens_.push_back(sampled_token);
    if (i + 1 == max_tokens) {
      RunDecodeStep({sampled_token}, cache_length_, false);
      break;
    }
    next = RunDecodeStep({sampled_token}, cache_length_, true);
  }

  std::string generated_text =
      generated_tokens.empty()
          ? std::string()
          : DecodeWithTokenizerJson(tokenizer_path, generated_tokens);
  generated_text =
      ApplyStopStrings(std::move(generated_text), combined_stop_strings);
  report_.generated_token_ids = generated_tokens;
  report_.generated_tokens = generated_tokens.size();
  chat_messages_.push_back({"user", user_text});
  chat_messages_.push_back({"assistant", generated_text});
  return generated_text;
#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON
}

std::vector<int64_t> OnnxLlmResidencyAdapter::GenerateContinuationTokens(
    const std::vector<int64_t>& tokens, int max_tokens) {
  return GenerateContinuationTokens(tokens, max_tokens, {});
}

std::vector<int64_t> OnnxLlmResidencyAdapter::GenerateContinuationTokens(
    const std::vector<int64_t>& tokens, int max_tokens,
    const std::vector<std::vector<int64_t>>& stop_token_sequences) {
  if (tokens.empty()) {
    throw std::runtime_error("ONNX LLM continuation tokens are required");
  }
  if (max_tokens <= 0) {
    throw std::runtime_error("max_tokens must be positive");
  }
  ValidateReadyForDecode();

  LogitsSampler sampler(options_.sampling);
  std::vector<int64_t> recent_tokens = tokens;
  DecodeResult next = RunDecodeStep(tokens, cache_length_, true);
  report_.generated_token_ids.clear();
  report_.generated_token_ids.reserve(static_cast<std::size_t>(max_tokens));
  for (int i = 0; i < max_tokens; ++i) {
    const int64_t sampled_token = sampler.Sample(next.logits, recent_tokens);
    report_.generated_token_ids.push_back(sampled_token);
    recent_tokens.push_back(sampled_token);
    if (RemoveStopSuffix(stop_token_sequences, &report_.generated_token_ids)) {
      break;
    }
    if (i + 1 == max_tokens) {
      break;
    }
    next = RunDecodeStep({sampled_token}, cache_length_, true);
  }

  report_.generated_tokens = report_.generated_token_ids.size();
  return report_.generated_token_ids;
}

void OnnxLlmResidencyAdapter::RestoreAndGenerateContinuation(
    const std::vector<int64_t>& tokens, int max_tokens,
    const std::vector<int64_t>& expected_tokens) {
  Timer timer;
  if (session_ == nullptr) {
    ReloadModel();
  }
  RestoreState();

  const std::vector<int64_t> generated =
      GenerateContinuationTokens(tokens, max_tokens);
  report_.generated_contains_expected =
      ContainsSubsequence(generated, expected_tokens);
  report_.restore_generate_ms = timer.ElapsedMs();
}

void OnnxLlmResidencyAdapter::CreateSession(double* elapsed_ms) {
  Timer timer;
  embedding_session_.reset();
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
  inputs_embeds_name_.clear();
  attention_mask_name_.clear();
  position_ids_name_.clear();
  position_ids_shape_.clear();
  num_logits_to_keep_name_.clear();
  logits_name_.clear();
  embedding_session_.reset();
  embedding_input_ids_name_.clear();
  embedding_output_name_.clear();
  inputs_embeds_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  embedding_input_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  embedding_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  num_logits_to_keep_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  embedding_hidden_size_ = 0;
  num_logits_to_keep_shape_.clear();
  kv_tensors_.clear();
  recurrent_tensors_.clear();
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
  std::map<std::string, Pair> recurrent_pairs;
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
    if (input_name == "inputs_embeds") {
      const ONNXTensorElementDataType element_type =
          tensor_info.GetElementType();
      if (!IsFloatTensor(element_type)) {
        std::ostringstream message;
        message << "ONNX LLM inputs_embeds has unsupported "
                << TensorElementTypeName(element_type) << " dtype";
        throw std::runtime_error(message.str());
      }
      inputs_embeds_name_ = input_name;
      inputs_embeds_element_type_ = element_type;
      embedding_hidden_size_ = KnownHiddenSize(tensor_info.GetShape());
      continue;
    }
    if (input_name == "attention_mask") {
      attention_mask_name_ = input_name;
      continue;
    }
    if (input_name == "position_ids") {
      position_ids_name_ = input_name;
      position_ids_shape_ = tensor_info.GetShape();
      report_.position_ids_present = true;
      continue;
    }
    if (input_name == "num_logits_to_keep") {
      const ONNXTensorElementDataType element_type =
          tensor_info.GetElementType();
      if (!IsIntTensor(element_type)) {
        std::ostringstream message;
        message << "ONNX LLM num_logits_to_keep has unsupported "
                << TensorElementTypeName(element_type) << " dtype";
        throw std::runtime_error(message.str());
      }
      num_logits_to_keep_name_ = input_name;
      num_logits_to_keep_element_type_ = element_type;
      num_logits_to_keep_shape_ = tensor_info.GetShape();
      continue;
    }

    int layer = 0;
    bool is_key = false;
    if (ParseKvName(input_name, "past_key_values.", &layer, &is_key)) {
      const ONNXTensorElementDataType element_type =
          tensor_info.GetElementType();
      if (!IsFloatTensor(element_type)) {
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

    std::string recurrent_suffix;
    std::string recurrent_key;
    if (ParsePrefixedStateName(input_name, "past_conv.", &recurrent_suffix)) {
      recurrent_key = "conv." + recurrent_suffix;
    } else if (ParsePrefixedStateName(input_name, "past_recurrent.",
                                      &recurrent_suffix)) {
      recurrent_key = "recurrent." + recurrent_suffix;
    }
    if (!recurrent_key.empty()) {
      const ONNXTensorElementDataType element_type =
          tensor_info.GetElementType();
      if (!IsFloatTensor(element_type)) {
        std::ostringstream message;
        message << "ONNX LLM recurrent input " << input_name
                << " has unsupported " << TensorElementTypeName(element_type)
                << " dtype";
        throw std::runtime_error(message.str());
      }
      Pair& pair = recurrent_pairs[recurrent_key];
      pair.past_name = input_name;
      pair.past_element_type = element_type;
      pair.shape = tensor_info.GetShape();
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
      if (!IsFloatTensor(logits_element_type_)) {
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
      if (!IsFloatTensor(element_type)) {
        std::ostringstream message;
        message << "ONNX LLM KV output " << output_name << " has unsupported "
                << TensorElementTypeName(element_type) << " dtype";
        throw std::runtime_error(message.str());
      }
      Pair& pair = kv_pairs[std::make_tuple(layer, is_key)];
      pair.present_name = output_name;
      pair.present_element_type = element_type;
      ++report_.present_output_count;
      continue;
    }

    std::string recurrent_suffix;
    std::string recurrent_key;
    if (ParsePrefixedStateName(output_name, "present_conv.",
                               &recurrent_suffix)) {
      recurrent_key = "conv." + recurrent_suffix;
    } else if (ParsePrefixedStateName(output_name, "present_recurrent.",
                                      &recurrent_suffix)) {
      recurrent_key = "recurrent." + recurrent_suffix;
    }
    if (!recurrent_key.empty()) {
      const ONNXTensorElementDataType element_type =
          tensor_info.GetElementType();
      if (!IsFloatTensor(element_type)) {
        std::ostringstream message;
        message << "ONNX LLM recurrent output " << output_name
                << " has unsupported " << TensorElementTypeName(element_type)
                << " dtype";
        throw std::runtime_error(message.str());
      }
      Pair& pair = recurrent_pairs[recurrent_key];
      pair.present_name = output_name;
      pair.present_element_type = element_type;
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

  for (const auto& [unused_key, pair] : recurrent_pairs) {
    (void)unused_key;
    if (pair.past_name.empty() || pair.present_name.empty()) {
      throw std::runtime_error(
          "ONNX LLM recurrent past/present pair is incomplete");
    }
    if (pair.past_element_type != pair.present_element_type) {
      throw std::runtime_error("ONNX LLM recurrent past/present dtypes differ");
    }
    RecurrentTensor tensor;
    tensor.past_name = pair.past_name;
    tensor.present_name = pair.present_name;
    tensor.element_type = pair.past_element_type;
    tensor.shape = StaticCacheShape(pair.shape);
    recurrent_tensors_.push_back(std::move(tensor));
  }

  if (input_ids_name_.empty() && !inputs_embeds_name_.empty()) {
    DiscoverEmbeddingModel();
  }

  report_.cache_surface_found =
      (!input_ids_name_.empty() || !inputs_embeds_name_.empty()) &&
      !attention_mask_name_.empty() && !logits_name_.empty() &&
      report_.past_input_count > 0 &&
      report_.present_output_count == report_.past_input_count;
  if (!report_.cache_surface_found) {
    throw std::runtime_error("ONNX LLM cache surface was not found");
  }
}

void OnnxLlmResidencyAdapter::DiscoverEmbeddingModel() {
  if (embedding_hidden_size_ <= 0) {
    throw std::runtime_error(
        "ONNX LLM inputs_embeds hidden size is unresolved");
  }

  const std::filesystem::path decoder_model_path(options_.model_path);
  const std::vector<std::filesystem::path> candidates =
      CandidateOnnxFiles(decoder_model_path);
  Ort::AllocatorWithDefaultOptions allocator;

  for (const std::filesystem::path& candidate : candidates) {
    try {
      const std::wstring candidate_path = ToWideString(candidate.string());
      auto candidate_session = std::make_unique<Ort::Session>(
          *env_, candidate_path.c_str(), *session_options_);

      std::string input_name;
      std::string output_name;
      ONNXTensorElementDataType input_element_type =
          ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      ONNXTensorElementDataType output_element_type =
          ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

      const std::size_t input_count = candidate_session->GetInputCount();
      bool unsupported_required_input = false;
      for (std::size_t i = 0; i < input_count; ++i) {
        Ort::AllocatedStringPtr name =
            candidate_session->GetInputNameAllocated(i, allocator);
        const std::string current_input_name = name.get();
        Ort::TypeInfo type_info = candidate_session->GetInputTypeInfo(i);
        Ort::ConstTensorTypeAndShapeInfo tensor_info =
            type_info.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> shape = tensor_info.GetShape();
        const ONNXTensorElementDataType element_type =
            tensor_info.GetElementType();

        if ((current_input_name == "input_ids" ||
             (input_name.empty() && IsIntTensor(element_type))) &&
            shape.size() == 2) {
          input_name = current_input_name;
          input_element_type = element_type;
          continue;
        }

        unsupported_required_input = true;
        break;
      }
      if (unsupported_required_input || input_name.empty()) {
        continue;
      }

      const std::size_t output_count = candidate_session->GetOutputCount();
      for (std::size_t i = 0; i < output_count; ++i) {
        Ort::AllocatedStringPtr name =
            candidate_session->GetOutputNameAllocated(i, allocator);
        const std::string current_output_name = name.get();
        Ort::TypeInfo type_info = candidate_session->GetOutputTypeInfo(i);
        Ort::ConstTensorTypeAndShapeInfo tensor_info =
            type_info.GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType element_type =
            tensor_info.GetElementType();
        const std::vector<int64_t> shape = tensor_info.GetShape();

        if (element_type == inputs_embeds_element_type_ &&
            ShapeCompatibleHiddenSize(shape, embedding_hidden_size_)) {
          output_name = current_output_name;
          output_element_type = element_type;
          break;
        }
      }
      if (output_name.empty()) {
        continue;
      }

      embedding_session_ = std::move(candidate_session);
      embedding_input_ids_name_ = std::move(input_name);
      embedding_output_name_ = std::move(output_name);
      embedding_input_element_type_ = input_element_type;
      embedding_element_type_ = output_element_type;
      return;
    } catch (const std::exception&) {
      continue;
    }
  }

  std::ostringstream message;
  message << "ONNX LLM decoder requires inputs_embeds, but no compatible "
          << "embedding ONNX model was found next to "
          << decoder_model_path.string();
  throw std::runtime_error(message.str());
}

OnnxLlmResidencyAdapter::EmbeddingResult
OnnxLlmResidencyAdapter::RunTokenEmbedding(
    const std::vector<int64_t>& input_tokens) {
  if (embedding_session_ == nullptr || embedding_input_ids_name_.empty() ||
      embedding_output_name_.empty() || embedding_hidden_size_ <= 0) {
    throw std::runtime_error("ONNX LLM embedding model is not ready");
  }

  const int64_t input_length = static_cast<int64_t>(input_tokens.size());
  const std::vector<int64_t> input_shape = {1, input_length};
  const std::vector<int64_t> output_shape = {1, input_length,
                                             embedding_hidden_size_};
  Ort::MemoryInfo cpu_memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::MemoryInfo cuda_memory_info("Cuda", OrtDeviceAllocator,
                                   options_.device_index, OrtMemTypeDefault);

  std::vector<Ort::Value> input_values;
  std::vector<Ort::Value> output_values;
  std::vector<int32_t> int32_tokens;
  input_values.reserve(1);
  output_values.reserve(1);

  if (embedding_input_element_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    input_values.push_back(Ort::Value::CreateTensor<int64_t>(
        cpu_memory_info, const_cast<int64_t*>(input_tokens.data()),
        input_tokens.size(), input_shape.data(), input_shape.size()));
  } else if (embedding_input_element_type_ ==
             ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    int32_tokens.reserve(input_tokens.size());
    for (const int64_t token : input_tokens) {
      if (token < std::numeric_limits<int32_t>::min() ||
          token > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(
            "ONNX LLM token id is outside int32 embedding range");
      }
      int32_tokens.push_back(static_cast<int32_t>(token));
    }
    input_values.push_back(Ort::Value::CreateTensor<int32_t>(
        cpu_memory_info, int32_tokens.data(), int32_tokens.size(),
        input_shape.data(), input_shape.size()));
  } else {
    throw std::runtime_error("ONNX LLM embedding input dtype is unsupported");
  }

  EmbeddingResult result;
  result.shape = output_shape;
  result.element_type = embedding_element_type_;
  result.buffer.Allocate(ElementCount(output_shape) *
                         TensorElementBytes(result.element_type));
  output_values.push_back(Ort::Value::CreateTensor(
      cuda_memory_info, result.buffer.data, result.buffer.bytes,
      result.shape.data(), result.shape.size(), result.element_type));

  Ort::IoBinding binding(*embedding_session_);
  binding.BindInput(embedding_input_ids_name_.c_str(), input_values.back());
  binding.BindOutput(embedding_output_name_.c_str(), output_values.back());
  embedding_session_->Run(Ort::RunOptions{nullptr}, binding);
  return result;
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
  for (RecurrentTensor& tensor : recurrent_tensors_) {
    tensor.buffer.Allocate(ElementCount(tensor.shape) *
                           TensorElementBytes(tensor.element_type));
    MOSAICVRAM_CUDA_CHECK(
        cudaMemset(tensor.buffer.data, 0, tensor.buffer.bytes),
        "initialize ONNX LLM recurrent cache");
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
  input_values.reserve(4 + kv_tensors_.size() + recurrent_tensors_.size());
  output_values.reserve(1 + kv_tensors_.size() + recurrent_tensors_.size());
  int64_t num_logits_to_keep64 = input_length;
  int32_t num_logits_to_keep32 = 0;

  const std::vector<int64_t> input_shape = {1, input_length};
  const std::vector<int64_t> mask_shape = {1, total_length};
  std::vector<int64_t> mask(static_cast<std::size_t>(total_length), 1);
  EmbeddingResult embedding_result;

  if (!input_ids_name_.empty()) {
    input_values.push_back(Ort::Value::CreateTensor<int64_t>(
        cpu_memory_info, const_cast<int64_t*>(input_tokens.data()),
        input_tokens.size(), input_shape.data(), input_shape.size()));
    binding.BindInput(input_ids_name_.c_str(), input_values.back());
  } else {
    embedding_result = RunTokenEmbedding(input_tokens);
    input_values.push_back(Ort::Value::CreateTensor(
        cuda_memory_info, embedding_result.buffer.data,
        embedding_result.buffer.bytes, embedding_result.shape.data(),
        embedding_result.shape.size(), embedding_result.element_type));
    binding.BindInput(inputs_embeds_name_.c_str(), input_values.back());
  }

  input_values.push_back(Ort::Value::CreateTensor<int64_t>(
      cpu_memory_info, mask.data(), mask.size(), mask_shape.data(),
      mask_shape.size()));
  binding.BindInput(attention_mask_name_.c_str(), input_values.back());

  if (!position_ids_name_.empty()) {
    std::vector<int64_t> position_shape = input_shape;
    int64_t position_planes = 1;
    if (position_ids_shape_.size() == 3) {
      position_planes = position_ids_shape_[0] > 0 ? position_ids_shape_[0] : 3;
      position_shape = {position_planes, 1, input_length};
    } else if (!position_ids_shape_.empty() &&
               position_ids_shape_.size() != input_shape.size()) {
      throw std::runtime_error("ONNX LLM position_ids shape is unsupported");
    }

    std::vector<int64_t> positions(
        static_cast<std::size_t>(position_planes * input_length));
    for (int64_t plane = 0; plane < position_planes; ++plane) {
      for (int64_t i = 0; i < input_length; ++i) {
        positions[static_cast<std::size_t>(plane * input_length + i)] =
            past_length + i;
      }
    }
    input_values.push_back(Ort::Value::CreateTensor<int64_t>(
        cpu_memory_info, positions.data(), positions.size(),
        position_shape.data(), position_shape.size()));
    binding.BindInput(position_ids_name_.c_str(), input_values.back());
  }

  if (!num_logits_to_keep_name_.empty()) {
    const std::vector<int64_t> default_scalar_shape = {};
    const std::vector<int64_t> default_vector_shape = {1};
    const std::vector<int64_t>* logits_to_keep_shape =
        num_logits_to_keep_shape_.empty() ? &default_scalar_shape
                                          : &num_logits_to_keep_shape_;
    for (const int64_t dim : *logits_to_keep_shape) {
      if (dim < 0) {
        logits_to_keep_shape = &default_vector_shape;
        break;
      }
    }
    if (num_logits_to_keep_element_type_ ==
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      input_values.push_back(Ort::Value::CreateTensor<int64_t>(
          cpu_memory_info, &num_logits_to_keep64, 1,
          logits_to_keep_shape->data(), logits_to_keep_shape->size()));
    } else if (num_logits_to_keep_element_type_ ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      num_logits_to_keep32 = static_cast<int32_t>(num_logits_to_keep64);
      input_values.push_back(Ort::Value::CreateTensor<int32_t>(
          cpu_memory_info, &num_logits_to_keep32, 1,
          logits_to_keep_shape->data(), logits_to_keep_shape->size()));
    } else {
      throw std::runtime_error(
          "ONNX LLM num_logits_to_keep dtype is unsupported");
    }
    binding.BindInput(num_logits_to_keep_name_.c_str(), input_values.back());
  }

  for (KvTensor& tensor : kv_tensors_) {
    input_values.push_back(Ort::Value::CreateTensor(
        cuda_memory_info, tensor.cache_buffer.data, tensor.cache_buffer.bytes,
        tensor.cache_shape.data(), tensor.cache_shape.size(),
        tensor.element_type));
    binding.BindInput(tensor.past_name.c_str(), input_values.back());
  }

  for (RecurrentTensor& tensor : recurrent_tensors_) {
    input_values.push_back(Ort::Value::CreateTensor(
        cuda_memory_info, tensor.buffer.data, tensor.buffer.bytes,
        tensor.shape.data(), tensor.shape.size(), tensor.element_type));
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

  std::vector<RecurrentTensor> present_recurrent_tensors;
  present_recurrent_tensors.reserve(recurrent_tensors_.size());
  for (const RecurrentTensor& tensor : recurrent_tensors_) {
    RecurrentTensor present;
    present.past_name = tensor.past_name;
    present.present_name = tensor.present_name;
    present.element_type = tensor.element_type;
    present.shape = tensor.shape;
    present.buffer.Allocate(ElementCount(present.shape) *
                            TensorElementBytes(present.element_type));
    output_values.push_back(Ort::Value::CreateTensor(
        cuda_memory_info, present.buffer.data, present.buffer.bytes,
        present.shape.data(), present.shape.size(), present.element_type));
    binding.BindOutput(present.present_name.c_str(), output_values.back());
    present_recurrent_tensors.push_back(std::move(present));
  }

  session_->Run(Ort::RunOptions{nullptr}, binding);

  const std::size_t logits_values =
      static_cast<std::size_t>(input_length * vocab_size_);
  const std::size_t last_token_offset =
      static_cast<std::size_t>((input_length - 1) * vocab_size_);
  double checksum = 0.0;
  std::vector<float> last_logits(static_cast<std::size_t>(vocab_size_));
  report_.logits_finite_count = 0;
  report_.logits_nan_count = 0;
  report_.logits_pos_inf_count = 0;
  report_.logits_neg_inf_count = 0;
  report_.decode_valid = false;

  if (!read_logits) {
    ReplaceCache(&present_tensors, &present_recurrent_tensors, total_length);
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
      last_logits[static_cast<std::size_t>(i)] = value;
      if (!std::isfinite(value)) {
        continue;
      }
      checksum += static_cast<double>(value);
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
      last_logits[static_cast<std::size_t>(i)] = value;
      if (!std::isfinite(value)) {
        continue;
      }
      checksum += static_cast<double>(value);
    }
  } else {
    throw std::runtime_error("ONNX LLM logits dtype is unsupported");
  }

  const int64_t best_token = GreedyToken(last_logits);
  if (best_token < 0) {
    throw std::runtime_error(DecodeFailureMessage(report_));
  }
  report_.decode_valid = true;

  ReplaceCache(&present_tensors, &present_recurrent_tensors, total_length);
  return DecodeResult{best_token, checksum, std::move(last_logits)};
}

void OnnxLlmResidencyAdapter::ReplaceCache(
    std::vector<KvTensor>* output_tensors,
    std::vector<RecurrentTensor>* output_recurrent_tensors,
    int64_t cache_length) {
  kv_tensors_ = std::move(*output_tensors);
  recurrent_tensors_ = std::move(*output_recurrent_tensors);
  cache_length_ = cache_length;
}

void OnnxLlmResidencyAdapter::FreeCache() {
  for (KvTensor& tensor : kv_tensors_) {
    tensor.cache_buffer.Free();
  }
  for (RecurrentTensor& tensor : recurrent_tensors_) {
    tensor.buffer.Free();
  }
  cache_length_ = 0;
}

void OnnxLlmResidencyAdapter::ValidateReadyForDecode() const {
  if (session_ == nullptr || kv_tensors_.empty() || vocab_size_ <= 0) {
    throw std::runtime_error("ONNX LLM adapter is not ready");
  }
  if (input_ids_name_.empty() &&
      (inputs_embeds_name_.empty() || embedding_session_ == nullptr)) {
    throw std::runtime_error("ONNX LLM embedding path is not ready");
  }
}

std::size_t OnnxLlmResidencyAdapter::CacheBytes() const {
  std::size_t bytes = 0;
  for (const KvTensor& tensor : kv_tensors_) {
    bytes += tensor.cache_buffer.bytes;
  }
  for (const RecurrentTensor& tensor : recurrent_tensors_) {
    bytes += tensor.buffer.bytes;
  }
  return bytes;
}

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_ONNX
