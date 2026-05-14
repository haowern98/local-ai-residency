#include "residency/llama_residency_adapter.h"

#ifdef MOSAICVRAM_ENABLE_LLAMA

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda/cuda_error.h"
#include "util/timer.h"

namespace mosaicvram {
namespace {

class ScopedLlamaBatch {
 public:
  explicit ScopedLlamaBatch(int32_t token_capacity)
      : batch_(llama_batch_init(token_capacity, /*embd=*/0,
                                /*n_seq_max=*/1)) {}

  ~ScopedLlamaBatch() { llama_batch_free(batch_); }

  ScopedLlamaBatch(const ScopedLlamaBatch&) = delete;
  ScopedLlamaBatch& operator=(const ScopedLlamaBatch&) = delete;

  llama_batch* get() { return &batch_; }

 private:
  llama_batch batch_;
};

void ClearBatch(llama_batch* batch) { batch->n_tokens = 0; }

void AddTokenToBatch(llama_batch* batch, llama_token token, llama_pos position,
                     llama_seq_id sequence_id, bool logits) {
  const int32_t index = batch->n_tokens;
  batch->token[index] = token;
  batch->pos[index] = position;
  batch->n_seq_id[index] = 1;
  batch->seq_id[index][0] = sequence_id;
  batch->logits[index] = logits;
  batch->n_tokens++;
}

bool IsBlank(std::string_view text) {
  for (char c : text) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  return true;
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

}  // namespace

LlamaResidencyAdapter::BackendLifetime::BackendLifetime() {
  llama_backend_init();
}

LlamaResidencyAdapter::BackendLifetime::~BackendLifetime() {
  llama_backend_free();
}

void LlamaResidencyAdapter::ModelDeleter::operator()(llama_model* model) const {
  if (model != nullptr) {
    llama_model_free(model);
  }
}

void LlamaResidencyAdapter::ContextDeleter::operator()(
    llama_context* context) const {
  if (context != nullptr) {
    llama_free(context);
  }
}

void LlamaResidencyAdapter::SamplerDeleter::operator()(
    llama_sampler* sampler) const {
  if (sampler != nullptr) {
    llama_sampler_free(sampler);
  }
}

LlamaResidencyAdapter::LlamaResidencyAdapter(LlamaResidencyOptions options)
    : options_(std::move(options)) {
  snapshot_.session_id = options_.session_id;
  snapshot_.backend_name = "llama.cpp";
}

LlamaResidencyAdapter::~LlamaResidencyAdapter() = default;

void LlamaResidencyAdapter::QuietLog(ggml_log_level level, const char* text,
                                     void* user_data) {
  (void)user_data;
  if (level >= GGML_LOG_LEVEL_ERROR && text != nullptr) {
    std::cerr << text;
  }
}

void LlamaResidencyAdapter::Load() {
  MOSAICVRAM_CUDA_CHECK(cudaSetDevice(options_.device_index),
                        "set llama CUDA device");
  llama_log_set(QuietLog, nullptr);

  Timer timer;
  model_ = LoadModel();
  report_.initial_model_load_ms = timer.ElapsedMs();
  vocab_ = llama_model_get_vocab(model_.get());
  residency_state_ = ResidencyState::kResident;
}

void LlamaResidencyAdapter::CreateContext() {
  if (model_ == nullptr) {
    throw std::runtime_error("cannot create llama context without model");
  }
  context_ = MakeContext();
  llama_sampler_chain_params sampler_params =
      llama_sampler_chain_default_params();
  chat_sampler_.reset(llama_sampler_chain_init(sampler_params));
  if (chat_sampler_ == nullptr) {
    throw std::runtime_error("failed to create llama sampler chain");
  }
  llama_sampler_chain_add(chat_sampler_.get(),
                          llama_sampler_init_min_p(options_.chat_min_p, 1));
  llama_sampler_chain_add(chat_sampler_.get(),
                          llama_sampler_init_temp(options_.chat_temperature));
  llama_sampler_chain_add(chat_sampler_.get(),
                          llama_sampler_init_dist(options_.chat_seed));
  ResetDecodePosition();
  chat_messages_.clear();
  chat_formatted_length_ = 0;
  residency_state_ = ResidencyState::kResident;
}

void LlamaResidencyAdapter::PrefillPrompt() {
  Timer timer;
  if (context_ == nullptr || vocab_ == nullptr) {
    throw std::runtime_error("llama context is not ready");
  }

  std::vector<llama_token> prompt_tokens = TokenizePrompt();
  if (prompt_tokens.size() >=
      static_cast<std::size_t>(options_.context_tokens - 2)) {
    throw std::runtime_error("prompt is too large for requested context");
  }
  report_.prompt_tokens = prompt_tokens.size();
  DecodeTokens(prompt_tokens);
  report_.first_token = GreedyToken();
  report_.prefill_ms = timer.ElapsedMs();
}

BackendStateSnapshot& LlamaResidencyAdapter::SaveState() {
  Timer timer;
  if (context_ == nullptr) {
    throw std::runtime_error("cannot save llama state without context");
  }

  const std::size_t full_size = llama_state_get_size(context_.get());
  snapshot_.full_state.Allocate(full_size);
  snapshot_.full_state_bytes = llama_state_get_data(
      context_.get(), snapshot_.full_state.data(), snapshot_.full_state.size());
  if (snapshot_.full_state_bytes == 0 ||
      snapshot_.full_state_bytes > snapshot_.full_state.size()) {
    throw std::runtime_error("failed to copy full llama state");
  }

  const std::size_t sequence_size =
      llama_state_seq_get_size(context_.get(), options_.sequence_id);
  snapshot_.sequence_state.Allocate(sequence_size);
  snapshot_.sequence_state_bytes = llama_state_seq_get_data(
      context_.get(), snapshot_.sequence_state.data(),
      snapshot_.sequence_state.size(), options_.sequence_id);
  if (snapshot_.sequence_state_bytes == 0 ||
      snapshot_.sequence_state_bytes > snapshot_.sequence_state.size()) {
    throw std::runtime_error("failed to copy llama sequence state");
  }

  snapshot_.source_residency = residency_state_;
  snapshot_decode_position_ = next_decode_position_;
  snapshot_chat_messages_ = chat_messages_;
  snapshot_chat_formatted_length_ = chat_formatted_length_;
  report_.full_state_bytes = snapshot_.full_state_bytes;
  report_.sequence_state_bytes = snapshot_.sequence_state_bytes;
  report_.save_state_ms = timer.ElapsedMs();
  return snapshot_;
}

void LlamaResidencyAdapter::CaptureBaselineNextToken() {
  std::vector<llama_token> continuation = {report_.first_token};
  DecodeTokens(continuation);
  report_.baseline_next_token = GreedyToken();
}

void LlamaResidencyAdapter::CheckFullRestore() {
  report_.restored_full_bytes = RestoreFullState();
  std::vector<llama_token> continuation = {report_.first_token};
  DecodeTokens(continuation);
  report_.full_restore_next_token = GreedyToken();
  report_.full_restore_match =
      report_.baseline_next_token == report_.full_restore_next_token;
}

void LlamaResidencyAdapter::CheckSequenceRestore() {
  report_.restored_sequence_bytes = RestoreSequenceState();
  std::vector<llama_token> continuation = {report_.first_token};
  DecodeTokens(continuation);
  report_.sequence_restore_next_token = GreedyToken();
  report_.sequence_restore_match =
      report_.baseline_next_token == report_.sequence_restore_next_token;
}

void LlamaResidencyAdapter::CheckSameContextClearAndRestore() {
  const void* same_context_address = context_.get();
  RestoreFullState();
  llama_memory_clear(llama_get_memory(context_.get()), /*data=*/true);
  report_.same_context_pointer_preserved =
      same_context_address == context_.get();
  report_.same_context_restore_bytes = RestoreFullState();
  std::vector<llama_token> continuation = {report_.first_token};
  DecodeTokens(continuation);
  report_.same_context_restore_next_token = GreedyToken();
  report_.same_context_restore_match =
      report_.baseline_next_token == report_.same_context_restore_next_token;
}

void LlamaResidencyAdapter::ResetConversation() {
  if (context_ != nullptr) {
    llama_memory_clear(llama_get_memory(context_.get()), /*data=*/true);
  }
  if (chat_sampler_ != nullptr) {
    llama_sampler_reset(chat_sampler_.get());
  }
  ResetDecodePosition();
  chat_messages_.clear();
  chat_formatted_length_ = 0;
}

void LlamaResidencyAdapter::EvictContext() {
  Timer timer;
  chat_sampler_.reset();
  context_.reset();
  residency_state_ = model_ == nullptr ? ResidencyState::kModelEvicted
                                       : ResidencyState::kContextEvicted;
  report_.evict_context_ms = timer.ElapsedMs();
}

void LlamaResidencyAdapter::RecreateContext() { CreateContext(); }

void LlamaResidencyAdapter::CheckRecreatedFullRestore() {
  report_.recreated_full_bytes = RestoreFullState();
  std::vector<llama_token> continuation = {report_.first_token};
  DecodeTokens(continuation);
  report_.recreated_full_next_token = GreedyToken();
  report_.recreated_full_restore_match =
      report_.baseline_next_token == report_.recreated_full_next_token;
}

void LlamaResidencyAdapter::CheckRecreatedSequenceRestore() {
  report_.recreated_sequence_bytes = RestoreSequenceState();
  std::vector<llama_token> continuation = {report_.first_token};
  DecodeTokens(continuation);
  report_.recreated_sequence_next_token = GreedyToken();
  report_.recreated_sequence_restore_match =
      report_.baseline_next_token == report_.recreated_sequence_next_token;
}

void LlamaResidencyAdapter::EvictModel() {
  Timer timer;
  chat_sampler_.reset();
  context_.reset();
  model_.reset();
  vocab_ = nullptr;
  residency_state_ = ResidencyState::kModelEvicted;
  report_.evict_model_ms = timer.ElapsedMs();
}

void LlamaResidencyAdapter::ReloadModel() {
  Timer timer;
  model_ = LoadModel();
  report_.model_reload_ms = timer.ElapsedMs();
  vocab_ = llama_model_get_vocab(model_.get());
  residency_state_ = ResidencyState::kContextEvicted;
}

void LlamaResidencyAdapter::RestoreState() {
  Timer timer;
  if (context_ == nullptr) {
    CreateContext();
  }
  report_.model_reloaded_full_bytes = RestoreFullState();
  report_.restore_state_ms = timer.ElapsedMs();
}

void LlamaResidencyAdapter::CheckModelReloadedFullRestore() {
  report_.model_reloaded_full_bytes = RestoreFullState();
  std::vector<llama_token> continuation = {report_.first_token};
  DecodeTokens(continuation);
  report_.model_reloaded_full_next_token = GreedyToken();
  report_.model_reloaded_full_restore_match =
      report_.baseline_next_token == report_.model_reloaded_full_next_token;
}

void LlamaResidencyAdapter::CheckModelReloadedSequenceRestore() {
  report_.model_reloaded_sequence_bytes = RestoreSequenceState();
  std::vector<llama_token> continuation = {report_.first_token};
  DecodeTokens(continuation);
  report_.model_reloaded_sequence_next_token = GreedyToken();
  report_.model_reloaded_sequence_restore_match =
      report_.baseline_next_token == report_.model_reloaded_sequence_next_token;
}

std::string LlamaResidencyAdapter::GenerateContinuation(const std::string& text,
                                                        int max_tokens) {
  if (max_tokens <= 0) {
    throw std::runtime_error("max_tokens must be positive");
  }
  if (context_ == nullptr) {
    throw std::runtime_error("cannot chat with llama session before load");
  }

  std::vector<llama_token> continuation = TokenizeText(text,
                                                       /*add_special=*/false);
  DecodeTokens(continuation);

  std::vector<llama_token> generated_tokens;
  generated_tokens.reserve(static_cast<std::size_t>(max_tokens));
  for (int i = 0; i < max_tokens; ++i) {
    const llama_token token = SampleToken();
    if (llama_vocab_is_eog(vocab_, token)) {
      break;
    }
    generated_tokens.push_back(token);
    std::vector<llama_token> next = {token};
    DecodeTokens(next);
  }

  report_.generated_tokens = generated_tokens.size();
  report_.generated_text = DetokenizeTokens(generated_tokens);
  return report_.generated_text;
}

std::string LlamaResidencyAdapter::GenerateChatReply(
    const std::string& user_text, int max_tokens) {
  if (max_tokens <= 0) {
    throw std::runtime_error("max_tokens must be positive");
  }
  if (context_ == nullptr) {
    throw std::runtime_error("cannot chat with llama session before load");
  }
  if (IsBlank(user_text)) {
    throw std::runtime_error("chat message is empty");
  }

  chat_messages_.push_back({"user", user_text});
  const std::string formatted = ApplyChatTemplate(/*add_assistant=*/true);
  if (chat_formatted_length_ < 0 ||
      chat_formatted_length_ > static_cast<int32_t>(formatted.size())) {
    throw std::runtime_error("invalid llama chat template cursor");
  }

  const std::string prompt_delta =
      formatted.substr(static_cast<std::size_t>(chat_formatted_length_));
  const bool is_first_chat_decode = next_decode_position_ == 0;
  std::vector<llama_token> prompt_tokens =
      TokenizeText(prompt_delta, is_first_chat_decode,
                   /*parse_special=*/true);
  DecodeTokens(prompt_tokens);

  std::vector<llama_token> generated_tokens;
  generated_tokens.reserve(static_cast<std::size_t>(max_tokens));
  std::string generated_text;
  for (int i = 0; i < max_tokens; ++i) {
    const llama_token token = SampleToken();
    if (llama_vocab_is_eog(vocab_, token)) {
      break;
    }
    generated_tokens.push_back(token);
    generated_text += DetokenizeTokens({token});
    const std::size_t stop =
        FindFirstStopString(generated_text, options_.stop_strings);
    if (stop != std::string::npos) {
      generated_text.resize(stop);
      break;
    }
    std::vector<llama_token> next = {token};
    DecodeTokens(next);
  }

  report_.generated_tokens = generated_tokens.size();
  report_.generated_text = generated_text;
  chat_messages_.push_back({"assistant", report_.generated_text});
  chat_formatted_length_ =
      static_cast<int32_t>(formatted.size() + report_.generated_text.size());
  return report_.generated_text;
}

void LlamaResidencyAdapter::RestoreAndGenerateContinuation(
    const std::string& text, int max_tokens, const std::string& expected_text) {
  Timer timer;
  if (context_ == nullptr) {
    CreateContext();
  }
  RestoreFullState();
  GenerateContinuation(text, max_tokens);
  report_.generated_contains_expected =
      !expected_text.empty() &&
      report_.generated_text.find(expected_text) != std::string::npos;
  report_.restore_generate_ms = timer.ElapsedMs();
}

bool LlamaResidencyAdapter::ResumeCheck() {
  Timer timer;
  if (!report_.model_reloaded_full_restore_match &&
      report_.baseline_next_token != LLAMA_TOKEN_NULL &&
      report_.first_token != LLAMA_TOKEN_NULL) {
    std::vector<llama_token> continuation = {report_.first_token};
    DecodeTokens(continuation);
    report_.model_reloaded_full_next_token = GreedyToken();
    report_.model_reloaded_full_restore_match =
        report_.baseline_next_token == report_.model_reloaded_full_next_token;
  }

  if (report_.full_restore_next_token == LLAMA_TOKEN_NULL &&
      report_.sequence_restore_next_token == LLAMA_TOKEN_NULL &&
      report_.recreated_full_next_token == LLAMA_TOKEN_NULL &&
      report_.recreated_sequence_next_token == LLAMA_TOKEN_NULL &&
      report_.same_context_restore_next_token == LLAMA_TOKEN_NULL &&
      report_.model_reloaded_sequence_next_token == LLAMA_TOKEN_NULL) {
    report_.resume_check_ms = timer.ElapsedMs();
    return report_.model_reloaded_full_restore_match;
  }

  const bool ok = report_.full_restore_match &&
                  report_.sequence_restore_match &&
                  report_.recreated_full_restore_match &&
                  report_.recreated_sequence_restore_match &&
                  report_.same_context_restore_match &&
                  report_.model_reloaded_full_restore_match &&
                  report_.model_reloaded_sequence_restore_match;
  report_.resume_check_ms = timer.ElapsedMs();
  return ok;
}

int LlamaResidencyAdapter::context_tokens() const {
  if (context_ == nullptr) {
    return 0;
  }
  return static_cast<int>(llama_n_ctx(context_.get()));
}

LlamaResidencyAdapter::ModelPtr LlamaResidencyAdapter::LoadModel() const {
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = options_.gpu_layers;
  model_params.main_gpu = options_.device_index;

  ModelPtr model(
      llama_model_load_from_file(options_.model_path.c_str(), model_params));
  if (model == nullptr) {
    throw std::runtime_error("failed to load llama.cpp model");
  }
  return model;
}

LlamaResidencyAdapter::ContextPtr LlamaResidencyAdapter::MakeContext() const {
  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = static_cast<uint32_t>(options_.context_tokens);
  context_params.n_batch = static_cast<uint32_t>(options_.batch_tokens);
  context_params.n_ubatch = static_cast<uint32_t>(options_.batch_tokens);
  context_params.n_threads = options_.threads;
  context_params.n_threads_batch = options_.threads;

  ContextPtr context(llama_init_from_model(model_.get(), context_params));
  if (context == nullptr) {
    throw std::runtime_error("failed to create llama.cpp context");
  }
  return context;
}

std::vector<llama_token> LlamaResidencyAdapter::TokenizePrompt() const {
  return TokenizeText(options_.prompt, /*add_special=*/true);
}

std::vector<llama_token> LlamaResidencyAdapter::TokenizeText(
    const std::string& text, bool add_special, bool parse_special) const {
  const int32_t text_size = static_cast<int32_t>(text.size());
  int32_t token_count = llama_tokenize(vocab_, text.c_str(), text_size, nullptr,
                                       0, add_special, parse_special);
  if (token_count == INT32_MIN) {
    throw std::runtime_error("llama tokenization overflow");
  }
  if (token_count < 0) {
    token_count = -token_count;
  }

  std::vector<llama_token> tokens(static_cast<std::size_t>(token_count));
  token_count = llama_tokenize(vocab_, text.c_str(), text_size, tokens.data(),
                               token_count, add_special, parse_special);
  if (token_count < 0) {
    throw std::runtime_error("llama tokenization failed");
  }
  tokens.resize(static_cast<std::size_t>(token_count));
  if (tokens.empty()) {
    throw std::runtime_error("llama tokenization produced no tokens");
  }
  return tokens;
}

void LlamaResidencyAdapter::DecodeTokens(
    const std::vector<llama_token>& tokens) {
  if (context_ == nullptr) {
    throw std::runtime_error("cannot decode llama tokens without context");
  }
  if (tokens.empty()) {
    return;
  }
  if (options_.batch_tokens <= 0) {
    throw std::runtime_error("llama batch size must be positive");
  }

  const llama_pos context_size =
      static_cast<llama_pos>(llama_n_ctx(context_.get()));
  if (next_decode_position_ + static_cast<llama_pos>(tokens.size()) >
      context_size) {
    throw std::runtime_error("llama tokens exceed configured context size");
  }

  const int32_t batch_capacity = static_cast<int32_t>(options_.batch_tokens);
  ScopedLlamaBatch scoped_batch(batch_capacity);
  llama_batch* batch = scoped_batch.get();

  std::size_t token_offset = 0;
  while (token_offset < tokens.size()) {
    ClearBatch(batch);
    while (batch->n_tokens < batch_capacity && token_offset < tokens.size()) {
      const bool logits = token_offset + 1 == tokens.size();
      AddTokenToBatch(batch, tokens[token_offset], next_decode_position_,
                      options_.sequence_id, logits);
      token_offset++;
      next_decode_position_++;
    }

    const int32_t status = llama_decode(context_.get(), *batch);
    if (status != 0) {
      throw std::runtime_error("llama_decode failed");
    }
  }
}

llama_token LlamaResidencyAdapter::GreedyToken() const {
  float* logits = llama_get_logits_ith(context_.get(), -1);
  if (logits == nullptr) {
    throw std::runtime_error("llama logits are unavailable");
  }

  const int32_t token_count = llama_vocab_n_tokens(vocab_);
  float best_logit = -std::numeric_limits<float>::infinity();
  llama_token best_token = LLAMA_TOKEN_NULL;
  for (int32_t i = 0; i < token_count; ++i) {
    if (logits[i] > best_logit) {
      best_logit = logits[i];
      best_token = i;
    }
  }
  if (best_token == LLAMA_TOKEN_NULL) {
    throw std::runtime_error("failed to select greedy llama token");
  }
  return best_token;
}

std::string LlamaResidencyAdapter::DetokenizeTokens(
    const std::vector<llama_token>& tokens) const {
  if (tokens.empty()) {
    return "";
  }
  std::string text;
  text.resize(tokens.size());
  int32_t chars = llama_detokenize(
      vocab_, tokens.data(), static_cast<int32_t>(tokens.size()), text.data(),
      static_cast<int32_t>(text.size()), /*remove_special=*/false,
      /*unparse_special=*/false);
  if (chars < 0) {
    text.resize(static_cast<std::size_t>(-chars));
    chars = llama_detokenize(
        vocab_, tokens.data(), static_cast<int32_t>(tokens.size()), text.data(),
        static_cast<int32_t>(text.size()),
        /*remove_special=*/false, /*unparse_special=*/false);
  }
  if (chars < 0 || chars > static_cast<int32_t>(text.size())) {
    throw std::runtime_error("llama detokenization failed");
  }
  text.resize(static_cast<std::size_t>(chars));
  return text;
}

llama_token LlamaResidencyAdapter::SampleToken() {
  if (context_ == nullptr || chat_sampler_ == nullptr) {
    throw std::runtime_error("llama sampler is unavailable");
  }
  const llama_token token =
      llama_sampler_sample(chat_sampler_.get(), context_.get(), -1);
  llama_sampler_accept(chat_sampler_.get(), token);
  return token;
}

std::string LlamaResidencyAdapter::ApplyChatTemplate(bool add_assistant) const {
  if (model_ == nullptr) {
    throw std::runtime_error("cannot apply llama chat template without model");
  }

  std::vector<llama_chat_message> messages;
  messages.reserve(chat_messages_.size());
  for (const LlamaChatMessage& message : chat_messages_) {
    messages.push_back({message.role.c_str(), message.content.c_str()});
  }

  const char* tmpl = nullptr;
  if (!options_.chat_template.empty()) {
    tmpl = options_.chat_template.c_str();
  } else {
    tmpl = llama_model_chat_template(model_.get(), nullptr);
  }

  int32_t length = llama_chat_apply_template(
      tmpl, messages.data(), messages.size(), add_assistant, nullptr, 0);
  if (length < 0) {
    throw std::runtime_error(
        "failed to apply llama chat template; set chat_template in "
        "sessions.txt");
  }
  std::string formatted(static_cast<std::size_t>(length), '\0');
  length = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                     add_assistant, formatted.data(),
                                     static_cast<int32_t>(formatted.size()));
  if (length < 0 || length > static_cast<int32_t>(formatted.size())) {
    throw std::runtime_error(
        "failed to apply llama chat template; set chat_template in "
        "sessions.txt");
  }
  formatted.resize(static_cast<std::size_t>(length));
  return formatted;
}

std::size_t LlamaResidencyAdapter::RestoreFullState() {
  if (snapshot_.full_state.empty()) {
    throw std::runtime_error("full llama state snapshot is empty");
  }
  const std::size_t read = llama_state_set_data(
      context_.get(), snapshot_.full_state.data(), snapshot_.full_state.size());
  if (read == 0 || read > snapshot_.full_state.size()) {
    throw std::runtime_error("failed to restore full llama state");
  }
  next_decode_position_ = snapshot_decode_position_;
  chat_messages_ = snapshot_chat_messages_;
  chat_formatted_length_ = snapshot_chat_formatted_length_;
  return read;
}

std::size_t LlamaResidencyAdapter::RestoreSequenceState() {
  if (snapshot_.sequence_state.empty()) {
    throw std::runtime_error("sequence llama state snapshot is empty");
  }
  llama_memory_clear(llama_get_memory(context_.get()), /*data=*/true);
  const std::size_t read = llama_state_seq_set_data(
      context_.get(), snapshot_.sequence_state.data(),
      snapshot_.sequence_state.size(), options_.sequence_id);
  if (read == 0 || read > snapshot_.sequence_state.size()) {
    throw std::runtime_error("failed to restore llama sequence state");
  }
  next_decode_position_ = snapshot_decode_position_;
  return read;
}

void LlamaResidencyAdapter::ResetDecodePosition() { next_decode_position_ = 0; }

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_LLAMA
