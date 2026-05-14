#include "commands/cli_command.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "residency/residency_controller.h"
#include "runtime/config_line.h"
#include "sampling/logits_sampler.h"

#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON
#include "tokenizer/tokenizers_cpp_adapter.h"
#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON

#ifdef MOSAICVRAM_ENABLE_LLAMA
#include "residency/llama_residency_adapter.h"
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
#include "residency/onnx_llm_residency_adapter.h"
#endif  // MOSAICVRAM_ENABLE_ONNX

#ifdef _WIN32
#include <windows.h>
#endif  // _WIN32

namespace mosaicvram {
namespace {

struct CliOptions {
  std::string config_path;
  bool help_requested = false;
};

struct SessionSpec {
  MosaicSessionId id = 0;
  std::string name;
  std::map<std::string, std::string> values;
  std::string issue;
};

struct CliSession {
  SessionSpec spec;
  std::unique_ptr<BackendStateAdapter> adapter;
};

bool ParseInt(std::string_view text, int* value) {
  int parsed = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }
  *value = parsed;
  return true;
}

bool ParseUint32(std::string_view text, uint32_t* value) {
  uint32_t parsed = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }
  *value = parsed;
  return true;
}

bool ParseFloat(std::string_view text, float* value) {
  std::string copy(text);
  char* end = nullptr;
  errno = 0;
  const float parsed = std::strtof(copy.c_str(), &end);
  if (errno != 0 || end == copy.c_str() || *end != '\0') {
    return false;
  }
  *value = parsed;
  return true;
}

std::string OptionalString(const SessionSpec& spec, const std::string& key,
                           const std::string& default_value) {
  const auto it = spec.values.find(key);
  return it == spec.values.end() ? default_value : it->second;
}

const std::string& RequiredValue(const SessionSpec& spec,
                                 const std::string& key) {
  const auto it = spec.values.find(key);
  if (it == spec.values.end() || it->second.empty()) {
    throw std::runtime_error("missing required key: " + key);
  }
  return it->second;
}

int OptionalInt(const SessionSpec& spec, const std::string& key,
                int default_value) {
  const auto it = spec.values.find(key);
  if (it == spec.values.end()) {
    return default_value;
  }
  int parsed = 0;
  if (!ParseInt(it->second, &parsed)) {
    throw std::runtime_error("invalid integer for " + key);
  }
  return parsed;
}

uint32_t OptionalUint32(const SessionSpec& spec, const std::string& key,
                        uint32_t default_value) {
  const auto it = spec.values.find(key);
  if (it == spec.values.end()) {
    return default_value;
  }
  uint32_t parsed = 0;
  if (!ParseUint32(it->second, &parsed)) {
    throw std::runtime_error("invalid unsigned integer for " + key);
  }
  return parsed;
}

float OptionalFloat(const SessionSpec& spec, const std::string& key,
                    float default_value) {
  const auto it = spec.values.find(key);
  if (it == spec.values.end()) {
    return default_value;
  }
  float parsed = 0.0f;
  if (!ParseFloat(it->second, &parsed)) {
    throw std::runtime_error("invalid float for " + key);
  }
  return parsed;
}

std::vector<std::string> SplitCommaSeparated(std::string_view text) {
  std::vector<std::string> values;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::size_t end =
        comma == std::string_view::npos ? text.size() : comma;
    std::string value = Trim(text.substr(start, end - start));
    if (!value.empty()) {
      values.push_back(std::move(value));
    }
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  return values;
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

void AppendMissingStrings(const std::vector<std::string>& values,
                          std::vector<std::string>* target) {
  for (const std::string& value : values) {
    if (std::find(target->begin(), target->end(), value) == target->end()) {
      target->push_back(value);
    }
  }
}

void ValidateSamplingOptions(const SamplingOptions& options) {
  if (options.temperature < 0.0f) {
    throw std::runtime_error("temp must be greater than or equal to zero");
  }
  if (options.top_k < 0) {
    throw std::runtime_error("top_k must be greater than or equal to zero");
  }
  if (options.top_p < 0.0f || options.top_p > 1.0f) {
    throw std::runtime_error("top_p must be between 0 and 1");
  }
  if (options.min_p < 0.0f || options.min_p > 1.0f) {
    throw std::runtime_error("min_p must be between 0 and 1");
  }
  if (options.repeat_penalty <= 0.0f) {
    throw std::runtime_error("repeat_penalty must be greater than zero");
  }
}

#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON
std::vector<std::vector<int64_t>> TokenizeStopStrings(
    const std::string& tokenizer_path,
    const std::vector<std::string>& stop_strings) {
  std::vector<std::vector<int64_t>> sequences;
  for (const std::string& stop : stop_strings) {
    std::vector<int64_t> tokens =
        TokenizeWithTokenizerJson(tokenizer_path, stop);
    if (!tokens.empty()) {
      sequences.push_back(std::move(tokens));
    }
  }
  return sequences;
}
#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON

void ValidateSessionSpec(SessionSpec* spec) {
  spec->issue.clear();
  const auto backend = spec->values.find("backend");
  if (backend == spec->values.end() || backend->second.empty()) {
    spec->issue = "missing required key: backend";
    return;
  }
  if (backend->second != "llama" && backend->second != "onnx-llm") {
    spec->issue = "unsupported backend: " + backend->second;
    return;
  }
  const auto model = spec->values.find("model");
  if (model == spec->values.end() || model->second.empty()) {
    spec->issue = "missing required key: model";
    return;
  }
  if (backend->second == "onnx-llm") {
    const auto tokenizer = spec->values.find("tokenizer");
    if (tokenizer == spec->values.end() || tokenizer->second.empty()) {
      spec->issue = "missing required key: tokenizer";
    }
  }
}

std::string ResidencyText(const CliSession& session) {
  if (session.adapter == nullptr) {
    return "Unloaded";
  }
  return ResidencyStateName(session.adapter->residency_state());
}

void PrintUsage() {
  std::cout << "Usage: mosaicvram.exe cli [--config <sessions.txt>]\n";
}

void PrintHelp() {
  std::cout << "Commands:\n"
            << "  /help\n"
            << "  /sessions\n"
            << "  /use <session>\n"
            << "  /chat [session]\n"
            << "  /load [session]\n"
            << "  /save [session]\n"
            << "  /evict [session]\n"
            << "  /restore [session]\n"
            << "  /reset <session>\n"
            << "  /reload-config\n"
            << "  /exit\n";
}

bool ParseOptions(int argc, char** argv, CliOptions* options) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options->help_requested = true;
      return true;
    }
    if (arg == "--config") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --config\n";
        return false;
      }
      options->config_path = argv[++i];
      continue;
    }
    std::cerr << "Unknown cli option: " << arg << "\n";
    return false;
  }
  if (options->config_path.empty()) {
    options->config_path = "sessions.txt";
  }
  options->config_path =
      std::filesystem::absolute(options->config_path).string();
  return true;
}

class CliRuntime {
 public:
  explicit CliRuntime(std::string config_path)
      : config_path_(std::move(config_path)),
        controller_(std::make_unique<ResidencyController>()) {}

  void ReloadConfig();
  void PrintSessions() const;
  void Use(const std::string& name);
  void EnterChat(const std::string& name);
  void Load(const std::string& name);
  void Save(const std::string& name);
  void Evict(const std::string& name);
  void Restore(const std::string& name);
  void Reset(const std::string& name);
  void ChatText(const std::string& text);

  std::string ResolveName(const std::vector<std::string>& positional) const;
  bool chat_mode() const { return chat_mode_; }
  const std::string& active_session() const { return active_session_; }
  const std::string& config_path() const { return config_path_; }

 private:
  CliSession* FindMutable(const std::string& name);
  const CliSession* Find(const std::string& name) const;
  void EnsureValid(const CliSession& session) const;
  void EnsureLoaded(const std::string& name, CliSession* session);
  void RegisterAdapter(CliSession* session);
  void RebuildController();

  std::map<std::string, CliSession> sessions_;
  std::string active_session_;
  std::string config_path_;
  bool chat_mode_ = false;
  MosaicSessionId next_session_id_ = 1;
  std::unique_ptr<ResidencyController> controller_;
};

void CliRuntime::ReloadConfig() {
  RebuildController();
  sessions_.clear();
  active_session_.clear();
  next_session_id_ = 1;
  chat_mode_ = false;

  std::ifstream input(config_path_);
  if (!input) {
    std::cout
        << "No sessions found.\n"
        << "Edit this file, then run /reload-config:\n"
        << "  " << config_path_ << "\n\n"
        << "Format:\n"
        << "  session llama backend=llama model=\"C:\\path\\to\\model.gguf\" "
           "ctx=16384 batch=512 gpu_layers=-1 device=0\n"
        << "  session onnx backend=onnx-llm "
           "model=\"C:\\path\\to\\model.onnx\" "
           "tokenizer=\"C:\\path\\to\\tokenizer_dir\" prefill_chunk=512 "
           "device=0\n";
    return;
  }

  std::string text;
  int line_number = 0;
  while (std::getline(input, text)) {
    ++line_number;
    const std::string trimmed = Trim(text);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }
    ConfigLine line = ParseConfigLine(trimmed, line_number);
    if (line.kind != "session" || line.positional.empty()) {
      std::cout << "ignored line " << line_number
                << ": expected session <name>\n";
      continue;
    }
    CliSession session;
    session.spec.id = next_session_id_++;
    session.spec.name = line.positional.front();
    session.spec.values = std::move(line.values);
    ValidateSessionSpec(&session.spec);
    if (sessions_.find(session.spec.name) != sessions_.end()) {
      std::cout << "ignored line " << line_number << ": duplicate session "
                << session.spec.name << "\n";
      continue;
    }
    sessions_.emplace(session.spec.name, std::move(session));
  }
  std::cout << "loaded config: " << config_path_ << "\n";
  PrintSessions();
}

void CliRuntime::PrintSessions() const {
  if (sessions_.empty()) {
    std::cout << "No sessions configured.\n";
    return;
  }
  std::cout << "session backend residency issue\n";
  for (const auto& [name, session] : sessions_) {
    const std::string marker =
        !active_session_.empty() && name == active_session_ ? "*" : " ";
    std::cout << marker << name << " "
              << OptionalString(session.spec, "backend", "-") << " "
              << ResidencyText(session) << " "
              << (session.spec.issue.empty() ? "-" : session.spec.issue)
              << "\n";
  }
}

void CliRuntime::Use(const std::string& name) {
  const CliSession* session = Find(name);
  EnsureValid(*session);
  active_session_ = name;
  std::cout << "active session: " << name << "\n";
}

void CliRuntime::EnterChat(const std::string& name) {
  const CliSession* session = Find(name);
  EnsureValid(*session);
  active_session_ = name;
  Load(name);
  chat_mode_ = true;
  std::cout << "chatting with " << name << "\n";
}

void CliRuntime::Load(const std::string& name) {
  CliSession* session = FindMutable(name);
  EnsureValid(*session);
  EnsureLoaded(name, session);
}

void CliRuntime::Save(const std::string& name) {
  CliSession* session = FindMutable(name);
  EnsureValid(*session);
  EnsureLoaded(name, session);
  const ResidencyControllerResult result =
      controller_->SaveSession(session->spec.id);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  std::cout << "saved " << name << "\n";
}

void CliRuntime::Evict(const std::string& name) {
  CliSession* session = FindMutable(name);
  EnsureValid(*session);
  if (session->adapter == nullptr) {
    chat_mode_ = false;
    std::cout << name << " is already unloaded\n";
    return;
  }
  const ResidencyControllerResult result =
      controller_->EvictModel(session->spec.id);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  chat_mode_ = false;
  std::cout << "evicted " << name << "\n";
}

void CliRuntime::Restore(const std::string& name) {
  CliSession* session = FindMutable(name);
  EnsureValid(*session);
  if (session->adapter == nullptr) {
    throw std::runtime_error("nothing to restore; use /load and /save first");
  }
  if (session->adapter->residency_state() == ResidencyState::kModelEvicted) {
    const ResidencyControllerResult reload_result =
        controller_->ReloadModel(session->spec.id);
    if (!reload_result.ok) {
      throw std::runtime_error(reload_result.error_message);
    }
  }
  const ResidencyControllerResult restore_result =
      controller_->RestoreSession(session->spec.id);
  if (!restore_result.ok) {
    throw std::runtime_error(restore_result.error_message);
  }
  chat_mode_ = true;
  std::cout << "restored " << name << "\n";
}

void CliRuntime::Reset(const std::string& name) {
  CliSession* session = FindMutable(name);
  EnsureValid(*session);
  const std::string backend = RequiredValue(session->spec, "backend");

  if (backend == "llama") {
#ifdef MOSAICVRAM_ENABLE_LLAMA
    if (session->adapter == nullptr) {
      std::cout << "reset " << name << "\n";
      return;
    }
    auto* adapter =
        dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("active session is not a llama session");
    }
    adapter->ResetConversation();
    std::cout << "reset " << name << "\n";
    return;
#else
    throw std::runtime_error(
        "CLI reset requires a build with llama.cpp enabled");
#endif  // MOSAICVRAM_ENABLE_LLAMA
  }

  if (backend == "onnx-llm") {
#ifdef MOSAICVRAM_ENABLE_ONNX
    if (session->adapter == nullptr) {
      std::cout << "reset " << name << "\n";
      return;
    }
    auto* adapter =
        dynamic_cast<OnnxLlmResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("active session is not an ONNX session");
    }
    adapter->ResetConversation();
    std::cout << "reset " << name << "\n";
    return;
#else
    throw std::runtime_error("CLI reset requires a build with ONNX enabled");
#endif  // MOSAICVRAM_ENABLE_ONNX
  }

  throw std::runtime_error("unsupported backend for reset: " + backend);
}

void CliRuntime::ChatText(const std::string& text) {
  const std::string name = ResolveName({});
  CliSession* session = FindMutable(name);
  EnsureValid(*session);
  const std::string backend = RequiredValue(session->spec, "backend");
  EnsureLoaded(name, session);

#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (backend == "llama") {
    auto* adapter =
        dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("active session is not a llama session");
    }
    const std::string generated =
        adapter->GenerateChatReply(text, adapter->options().chat_max_tokens);
    std::cout << name << ": " << generated << "\n";
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
  if (backend == "onnx-llm") {
    auto* adapter =
        dynamic_cast<OnnxLlmResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("active session is not an ONNX session");
    }
#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON
    const std::string& tokenizer_path =
        RequiredValue(session->spec, "tokenizer");
    const int max_tokens = OptionalInt(session->spec, "max_tokens", 512);
    if (max_tokens <= 0) {
      throw std::runtime_error("max_tokens must be greater than zero");
    }
    const TokenizerChatPrompt chat_prompt =
        ApplyTokenizerChatTemplate(tokenizer_path, text);
    const std::vector<int64_t> input_tokens =
        TokenizeWithTokenizerJson(tokenizer_path, chat_prompt.text);
    std::vector<std::string> stop_strings =
        SplitCommaSeparated(OptionalString(session->spec, "stop_strings", ""));
    AppendMissingStrings(chat_prompt.stop_strings, &stop_strings);
    const std::vector<std::vector<int64_t>> stop_token_sequences =
        TokenizeStopStrings(tokenizer_path, stop_strings);
    const std::vector<int64_t> generated_tokens =
        adapter->GenerateContinuationTokens(input_tokens, max_tokens,
                                            stop_token_sequences);
    const std::string generated = ApplyStopStrings(
        DecodeWithTokenizerJson(tokenizer_path, generated_tokens),
        stop_strings);
    std::cout << name << ": " << generated << "\n";
    return;
#else
    throw std::runtime_error(
        "ONNX CLI chat requires tokenizer.json support in this build");
#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON
  }
#endif  // MOSAICVRAM_ENABLE_ONNX

  throw std::runtime_error("unsupported backend for chat: " + backend);
}

std::string CliRuntime::ResolveName(
    const std::vector<std::string>& positional) const {
  if (!positional.empty()) {
    return positional.front();
  }
  if (active_session_.empty()) {
    throw std::runtime_error("no active session; use /use <session>");
  }
  return active_session_;
}

CliSession* CliRuntime::FindMutable(const std::string& name) {
  auto it = sessions_.find(name);
  if (it == sessions_.end()) {
    throw std::runtime_error("unknown session: " + name);
  }
  return &it->second;
}

const CliSession* CliRuntime::Find(const std::string& name) const {
  auto it = sessions_.find(name);
  if (it == sessions_.end()) {
    throw std::runtime_error("unknown session: " + name);
  }
  return &it->second;
}

void CliRuntime::EnsureValid(const CliSession& session) const {
  if (!session.spec.issue.empty()) {
    throw std::runtime_error("invalid session " + session.spec.name + ": " +
                             session.spec.issue);
  }
}

void CliRuntime::EnsureLoaded(const std::string& name, CliSession* session) {
  if (session->adapter == nullptr) {
    RegisterAdapter(session);
  }
  const ResidencyState state = session->adapter->residency_state();
  const std::string backend = RequiredValue(session->spec, "backend");
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (backend == "llama") {
    auto* adapter =
        dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("llama session has invalid adapter");
    }
    if (state == ResidencyState::kUnloaded) {
      adapter->Load();
      adapter->CreateContext();
      std::cout << "loaded " << name << "\n";
      return;
    }
    if (state == ResidencyState::kModelEvicted) {
      adapter->ReloadModel();
      adapter->CreateContext();
      std::cout << "loaded " << name << "\n";
      return;
    }
    if (state == ResidencyState::kContextEvicted) {
      adapter->CreateContext();
      std::cout << "loaded " << name << "\n";
    }
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
  if (backend == "onnx-llm") {
    auto* adapter =
        dynamic_cast<OnnxLlmResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("onnx-llm session has invalid adapter");
    }
    if (state == ResidencyState::kUnloaded ||
        state == ResidencyState::kModelEvicted ||
        state == ResidencyState::kContextEvicted) {
      adapter->Load();
      std::cout << "loaded " << name << "\n";
    }
    return;
  }
#endif  // MOSAICVRAM_ENABLE_ONNX

  throw std::runtime_error("unsupported backend: " + backend);
}

void CliRuntime::RegisterAdapter(CliSession* session) {
  const std::string backend = RequiredValue(session->spec, "backend");
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (backend == "llama") {
    LlamaResidencyOptions options;
    options.session_id = session->spec.id;
    options.model_path = RequiredValue(session->spec, "model");
    options.context_tokens =
        OptionalInt(session->spec, "ctx", options.context_tokens);
    options.batch_tokens =
        OptionalInt(session->spec, "batch", options.batch_tokens);
    options.gpu_layers =
        OptionalInt(session->spec, "gpu_layers", options.gpu_layers);
    options.device_index =
        OptionalInt(session->spec, "device", options.device_index);
    options.threads = OptionalInt(session->spec, "threads", options.threads);
    options.chat_max_tokens =
        OptionalInt(session->spec, "max_tokens", options.chat_max_tokens);
    options.chat_temperature =
        OptionalFloat(session->spec, "temp", options.chat_temperature);
    options.chat_min_p =
        OptionalFloat(session->spec, "min_p", options.chat_min_p);
    options.chat_seed =
        OptionalUint32(session->spec, "seed", options.chat_seed);
    options.chat_template =
        OptionalString(session->spec, "chat_template", options.chat_template);
    options.stop_strings =
        SplitCommaSeparated(OptionalString(session->spec, "stop_strings", ""));
    if (options.chat_max_tokens <= 0) {
      throw std::runtime_error("max_tokens must be greater than zero");
    }
    if (options.chat_temperature < 0.0f) {
      throw std::runtime_error("temp must be greater than or equal to zero");
    }
    if (options.chat_min_p < 0.0f || options.chat_min_p > 1.0f) {
      throw std::runtime_error("min_p must be between 0 and 1");
    }
    options.sequence_id =
        OptionalInt(session->spec, "seq_id", options.sequence_id);
    session->adapter = std::make_unique<LlamaResidencyAdapter>(options);
    const ResidencyControllerResult result =
        controller_->RegisterAdapter(session->adapter.get());
    if (!result.ok) {
      throw std::runtime_error(result.error_message);
    }
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
  if (backend == "onnx-llm") {
    OnnxLlmResidencyOptions options;
    options.session_id = session->spec.id;
    options.model_path = RequiredValue(session->spec, "model");
    options.prefill_chunk_tokens = OptionalInt(session->spec, "prefill_chunk",
                                               options.prefill_chunk_tokens);
    options.device_index =
        OptionalInt(session->spec, "device", options.device_index);
    options.sampling.temperature =
        OptionalFloat(session->spec, "temp", options.sampling.temperature);
    options.sampling.top_k =
        OptionalInt(session->spec, "top_k", options.sampling.top_k);
    options.sampling.top_p =
        OptionalFloat(session->spec, "top_p", options.sampling.top_p);
    options.sampling.min_p =
        OptionalFloat(session->spec, "min_p", options.sampling.min_p);
    options.sampling.repeat_penalty = OptionalFloat(
        session->spec, "repeat_penalty", options.sampling.repeat_penalty);
    options.sampling.seed =
        OptionalUint32(session->spec, "seed", options.sampling.seed);
    ValidateSamplingOptions(options.sampling);
    session->adapter = std::make_unique<OnnxLlmResidencyAdapter>(options);
    const ResidencyControllerResult result =
        controller_->RegisterAdapter(session->adapter.get());
    if (!result.ok) {
      throw std::runtime_error(result.error_message);
    }
    return;
  }
#endif  // MOSAICVRAM_ENABLE_ONNX

  throw std::runtime_error("unsupported backend in this build: " + backend);
}

void CliRuntime::RebuildController() {
  controller_ = std::make_unique<ResidencyController>();
}

void ExecuteCommand(const ConfigLine& line, CliRuntime* runtime,
                    bool* exit_requested) {
  if (line.kind == "/help") {
    PrintHelp();
    return;
  }
  if (line.kind == "/exit" || line.kind == "/quit") {
    *exit_requested = true;
    return;
  }
  if (line.kind == "/sessions") {
    runtime->PrintSessions();
    return;
  }
  if (line.kind == "/reload-config") {
    runtime->ReloadConfig();
    return;
  }
  if (line.kind == "/use") {
    runtime->Use(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "/chat") {
    runtime->EnterChat(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "/load") {
    runtime->Load(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "/save") {
    runtime->Save(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "/evict") {
    runtime->Evict(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "/restore") {
    runtime->Restore(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "/reset") {
    if (line.positional.empty()) {
      throw std::runtime_error(
          "missing session argument; use /reset <session>");
    }
    runtime->Reset(line.positional.front());
    return;
  }
  throw std::runtime_error("unknown command: " + line.kind);
}

void ConfigureCliConsole() {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif  // _WIN32
}

}  // namespace

int RunCliCommand(int argc, char** argv) {
  CliOptions options;
  if (!ParseOptions(argc, argv, &options)) {
    return EXIT_FAILURE;
  }
  if (options.help_requested) {
    PrintUsage();
    return EXIT_SUCCESS;
  }

  ConfigureCliConsole();

  CliRuntime runtime(options.config_path);
  std::cout << "MosaicVRAM CLI\n"
            << "Config: " << runtime.config_path() << "\n";
  runtime.ReloadConfig();

  bool exit_requested = false;
  std::string line_text;
  while (!exit_requested) {
    const std::string prompt =
        runtime.chat_mode() && !runtime.active_session().empty()
            ? runtime.active_session() + "> "
            : "mosaic> ";
    std::cout << prompt;
    if (!std::getline(std::cin, line_text)) {
      break;
    }
    const std::string trimmed = Trim(line_text);
    if (trimmed.empty()) {
      continue;
    }
    try {
      if (trimmed.front() == '/') {
        ExecuteCommand(ParseConfigLine(trimmed, 0), &runtime, &exit_requested);
      } else if (runtime.chat_mode()) {
        runtime.ChatText(trimmed);
      } else {
        std::cout << "error: enter /chat first\n";
      }
    } catch (const std::exception& error) {
      std::cout << "error: " << error.what() << "\n";
    }
  }
  return EXIT_SUCCESS;
}

}  // namespace mosaicvram
