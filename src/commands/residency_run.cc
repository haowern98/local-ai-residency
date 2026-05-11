#include "commands/residency_run.h"

#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstdlib>
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

#include "reload/reload_mode.h"
#include "residency/backend_state_adapter.h"
#include "residency/residency_controller.h"

#ifdef MOSAICVRAM_ENABLE_LLAMA
#include "residency/llama_residency_adapter.h"
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
#include "residency/onnx_llm_residency_adapter.h"
#endif  // MOSAICVRAM_ENABLE_ONNX

namespace mosaicvram {
namespace {

struct PlanLine {
  std::string kind;
  std::map<std::string, std::string> values;
  int line_number = 0;
};

struct SessionRuntime {
  std::string backend;
  std::unique_ptr<BackendStateAdapter> adapter;
};

struct Options {
  std::string plan_path;
  bool help_requested = false;
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

bool ParseUint64(std::string_view text, std::uint64_t* value) {
  std::uint64_t parsed = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }
  *value = parsed;
  return true;
}

bool ParseInt64(std::string_view text, int64_t* value) {
  int64_t parsed = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }
  *value = parsed;
  return true;
}

std::string Trim(std::string_view text) {
  std::size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }
  std::size_t end = text.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

std::vector<std::string> TokenizePlanLine(const std::string& line) {
  std::vector<std::string> tokens;
  std::string current;
  bool in_quotes = false;
  for (char c : line) {
    if (c == '"') {
      in_quotes = !in_quotes;
      continue;
    }
    if (!in_quotes && c == '#') {
      break;
    }
    if (!in_quotes && std::isspace(static_cast<unsigned char>(c))) {
      if (!current.empty()) {
        tokens.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(c);
  }
  if (!current.empty()) {
    tokens.push_back(current);
  }
  return tokens;
}

const std::string& RequiredValue(const PlanLine& line, const std::string& key) {
  const auto it = line.values.find(key);
  if (it == line.values.end() || it->second.empty()) {
    std::ostringstream message;
    message << "line " << line.line_number << " missing required key: " << key;
    throw std::runtime_error(message.str());
  }
  return it->second;
}

std::string BoolText(bool value) { return value ? "yes" : "no"; }

std::string EscapeReportValue(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '\n') {
      escaped += "\\n";
    } else if (c == '\r') {
      escaped += "\\r";
    } else if (c == '\t') {
      escaped += "\\t";
    } else {
      escaped.push_back(c);
    }
  }
  return escaped;
}

std::string TokenListText(const std::vector<int64_t>& tokens) {
  std::ostringstream text;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      text << ",";
    }
    text << tokens[i];
  }
  return text.str();
}

int OptionalInt(const PlanLine& line, const std::string& key,
                int default_value) {
  const auto it = line.values.find(key);
  if (it == line.values.end()) {
    return default_value;
  }
  int parsed = 0;
  if (!ParseInt(it->second, &parsed)) {
    std::ostringstream message;
    message << "line " << line.line_number << " has invalid integer for "
            << key;
    throw std::runtime_error(message.str());
  }
  return parsed;
}

std::string OptionalString(const PlanLine& line, const std::string& key,
                           const std::string& default_value) {
  const auto it = line.values.find(key);
  if (it == line.values.end()) {
    return default_value;
  }
  return it->second;
}

ReloadPolicy ParseReloadPolicy(const PlanLine& line) {
  ReloadPolicy policy;
  const std::string mode_str = OptionalString(line, "reload_mode", "cold");
  if (mode_str == "mmap") {
    policy.mode = ReloadMode::kMmap;
  } else if (mode_str == "streamed_vram") {
    policy.mode = ReloadMode::kStreamedVram;
  }
  policy.ram_budget_mb =
      static_cast<std::size_t>(OptionalInt(line, "ram_budget_mb", 0));
  policy.prefetch_mb =
      static_cast<std::size_t>(OptionalInt(line, "prefetch_mb", 0));
  policy.pack_path = OptionalString(line, "pack_path", "");
  const std::string create_pack_str =
      OptionalString(line, "create_pack", "no");
  policy.create_pack = create_pack_str == "yes";
  return policy;
}

MosaicSessionId RequiredSessionId(const PlanLine& line,
                                  const std::string& key) {
  std::uint64_t parsed = 0;
  if (!ParseUint64(RequiredValue(line, key), &parsed) || parsed == 0) {
    std::ostringstream message;
    message << "line " << line.line_number << " has invalid session id";
    throw std::runtime_error(message.str());
  }
  return static_cast<MosaicSessionId>(parsed);
}

std::vector<int64_t> ParseTokenList(const std::string& text, int line_number) {
  std::vector<int64_t> tokens;
  std::string_view rest = text;
  while (!rest.empty()) {
    const std::size_t comma_pos = rest.find(',');
    const std::string_view item =
        comma_pos == std::string_view::npos ? rest : rest.substr(0, comma_pos);
    int64_t token = 0;
    if (!ParseInt64(item, &token) || token < 0) {
      std::ostringstream message;
      message << "line " << line_number << " has invalid token list";
      throw std::runtime_error(message.str());
    }
    tokens.push_back(token);
    if (comma_pos == std::string_view::npos) {
      break;
    }
    rest = rest.substr(comma_pos + 1);
  }
  if (tokens.empty()) {
    std::ostringstream message;
    message << "line " << line_number << " has empty token list";
    throw std::runtime_error(message.str());
  }
  return tokens;
}

std::vector<int64_t> OptionalTokenList(const PlanLine& line,
                                       const std::string& key) {
  const auto it = line.values.find(key);
  if (it == line.values.end() || it->second.empty()) {
    return {};
  }
  return ParseTokenList(it->second, line.line_number);
}

std::vector<PlanLine> LoadPlan(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open plan file: " + path);
  }

  std::vector<PlanLine> lines;
  std::string text;
  int line_number = 0;
  while (std::getline(input, text)) {
    ++line_number;
    const std::string trimmed = Trim(text);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }

    const std::vector<std::string> tokens = TokenizePlanLine(trimmed);
    if (tokens.empty()) {
      continue;
    }

    PlanLine line;
    line.kind = tokens.front();
    line.line_number = line_number;
    for (std::size_t i = 1; i < tokens.size(); ++i) {
      const std::size_t equals_pos = tokens[i].find('=');
      if (equals_pos == std::string::npos || equals_pos == 0 ||
          equals_pos + 1 >= tokens[i].size()) {
        std::ostringstream message;
        message << "line " << line_number << " has invalid key=value token";
        throw std::runtime_error(message.str());
      }
      line.values[tokens[i].substr(0, equals_pos)] =
          tokens[i].substr(equals_pos + 1);
    }
    lines.push_back(std::move(line));
  }
  return lines;
}

void PrintUsage() {
  std::cout << "Usage: mosaicvram.exe residency-run --plan <path>\n";
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options->help_requested = true;
      return true;
    }
    if (i + 1 >= argc) {
      std::cerr << "Missing value for option: " << arg << "\n";
      return false;
    }
    const std::string_view value = argv[++i];
    if (arg == "--plan") {
      options->plan_path = std::string(value);
    } else {
      std::cerr << "Unknown residency-run option: " << arg << "\n";
      return false;
    }
  }
  if (options->plan_path.empty()) {
    std::cerr << "--plan is required\n";
    return false;
  }
  return true;
}

std::unique_ptr<BackendStateAdapter> CreateAdapter(const PlanLine& line) {
  const std::string& backend = RequiredValue(line, "backend");
  const MosaicSessionId session_id = RequiredSessionId(line, "id");
  (void)session_id;

#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (backend == "llama") {
    LlamaResidencyOptions options;
    options.session_id = session_id;
    options.model_path = RequiredValue(line, "model");
    options.prompt = RequiredValue(line, "prompt");
    options.context_tokens = OptionalInt(line, "ctx", options.context_tokens);
    options.batch_tokens = OptionalInt(line, "batch", options.batch_tokens);
    options.gpu_layers = OptionalInt(line, "gpu_layers", options.gpu_layers);
    options.device_index = OptionalInt(line, "device", options.device_index);
    options.threads = OptionalInt(line, "threads", options.threads);
    options.sequence_id = OptionalInt(line, "seq_id", options.sequence_id);
    auto adapter =
        std::make_unique<LlamaResidencyAdapter>(std::move(options));
    adapter->set_reload_policy(ParseReloadPolicy(line));
    return adapter;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
  if (backend == "onnx-llm") {
    OnnxLlmResidencyOptions options;
    options.session_id = session_id;
    options.model_path = RequiredValue(line, "model");
    options.prompt_tokens =
        ParseTokenList(RequiredValue(line, "tokens"), line.line_number);
    options.prefill_chunk_tokens =
        OptionalInt(line, "prefill_chunk", options.prefill_chunk_tokens);
    options.device_index = OptionalInt(line, "device", options.device_index);
    auto adapter =
        std::make_unique<OnnxLlmResidencyAdapter>(std::move(options));
    adapter->set_reload_policy(ParseReloadPolicy(line));
    return adapter;
  }
#endif  // MOSAICVRAM_ENABLE_ONNX

  std::ostringstream message;
  message << "line " << line.line_number
          << " uses unsupported backend: " << backend;
  throw std::runtime_error(message.str());
}

void LoadSession(SessionRuntime* session) {
  (void)session;
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (session->backend == "llama") {
    auto* adapter =
        dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("llama session has invalid adapter");
    }
    adapter->Load();
    adapter->CreateContext();
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
  if (session->backend == "onnx-llm") {
    auto* adapter =
        dynamic_cast<OnnxLlmResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("onnx-llm session has invalid adapter");
    }
    adapter->Load();
    return;
  }
#endif  // MOSAICVRAM_ENABLE_ONNX

  throw std::runtime_error("unsupported backend for load");
}

void PrefillSession(SessionRuntime* session) {
  (void)session;
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (session->backend == "llama") {
    auto* adapter =
        dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("llama session has invalid adapter");
    }
    adapter->PrefillPrompt();
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
  if (session->backend == "onnx-llm") {
    auto* adapter =
        dynamic_cast<OnnxLlmResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("onnx-llm session has invalid adapter");
    }
    adapter->PrefillPrompt();
    return;
  }
#endif  // MOSAICVRAM_ENABLE_ONNX

  throw std::runtime_error("unsupported backend for prefill");
}

void CaptureBaseline(SessionRuntime* session) {
  (void)session;
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (session->backend == "llama") {
    auto* adapter =
        dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("llama session has invalid adapter");
    }
    adapter->CaptureBaselineNextToken();
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
  if (session->backend == "onnx-llm") {
    auto* adapter =
        dynamic_cast<OnnxLlmResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("onnx-llm session has invalid adapter");
    }
    adapter->CaptureBaselineNextToken();
    return;
  }
#endif  // MOSAICVRAM_ENABLE_ONNX

  throw std::runtime_error("unsupported backend for baseline");
}

void RestoreThenGenerate(SessionRuntime* session, const PlanLine& line) {
  (void)session;
  (void)line;
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (session->backend == "llama") {
    auto* adapter =
        dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("llama session has invalid adapter");
    }
    adapter->RestoreAndGenerateContinuation(
        RequiredValue(line, "text"),
        OptionalInt(line, "max_tokens", 32),
        OptionalString(line, "expect", ""));
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
  if (session->backend == "onnx-llm") {
    auto* adapter =
        dynamic_cast<OnnxLlmResidencyAdapter*>(session->adapter.get());
    if (adapter == nullptr) {
      throw std::runtime_error("onnx-llm session has invalid adapter");
    }
    adapter->RestoreAndGenerateContinuation(
        ParseTokenList(RequiredValue(line, "tokens"), line.line_number),
        OptionalInt(line, "max_tokens", 32),
        OptionalTokenList(line, "expect_tokens"));
    return;
  }
#endif  // MOSAICVRAM_ENABLE_ONNX

  throw std::runtime_error(
      "restore_then_generate requires llama or onnx-llm");
}

void PrintAdapterReport(MosaicSessionId session_id,
                        const SessionRuntime& session) {
  (void)session_id;
  (void)session;
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (session.backend == "llama") {
    const auto* adapter =
        dynamic_cast<const LlamaResidencyAdapter*>(session.adapter.get());
    if (adapter == nullptr) {
      return;
    }
    const LlamaResidencyReport& report = adapter->report();
    std::cout << "session" << session_id
              << "_prompt_tokens=" << report.prompt_tokens << "\n"
              << "session" << session_id
              << "_reload_mode=" << ReloadModeName(adapter->reload_mode())
              << "\n"
              << "session" << session_id
              << "_full_state_bytes=" << report.full_state_bytes << "\n"
              << "session" << session_id
              << "_baseline_next_token=" << report.baseline_next_token << "\n"
              << "session" << session_id << "_restored_next_token="
              << report.model_reloaded_full_next_token << "\n"
              << "session" << session_id << "_resume_match="
              << BoolText(report.model_reloaded_full_restore_match) << "\n"
              << "session" << session_id
              << "_initial_model_load_ms=" << report.initial_model_load_ms
              << "\n"
              << "session" << session_id
              << "_model_reload_ms=" << report.model_reload_ms << "\n"
              << "session" << session_id
              << "_reload_file_open_ms=" << report.reload_file_open_ms << "\n"
              << "session" << session_id
              << "_reload_header_parse_ms=" << report.reload_header_parse_ms
              << "\n"
              << "session" << session_id
              << "_reload_tensor_plan_ms=" << report.reload_tensor_plan_ms
              << "\n"
              << "session" << session_id
              << "_reload_h2d_copy_ms=" << report.reload_h2d_copy_ms << "\n"
              << "session" << session_id
              << "_vram_before_load_mb="
              << (report.vram_before_load_bytes / (1024 * 1024)) << "\n"
              << "session" << session_id
              << "_vram_after_load_mb="
              << (report.vram_after_load_bytes / (1024 * 1024)) << "\n"
              << "session" << session_id
              << "_vram_after_evict_model_mb="
              << (report.vram_after_evict_model_bytes / (1024 * 1024)) << "\n"
              << "session" << session_id
              << "_vram_after_reload_mb="
              << (report.vram_after_reload_bytes / (1024 * 1024)) << "\n"
              << "session" << session_id
              << "_ram_before_reload_mb="
              << (report.ram_before_reload_bytes / (1024 * 1024)) << "\n"
              << "session" << session_id
              << "_ram_peak_during_reload_mb="
              << (report.ram_peak_during_reload_bytes / (1024 * 1024)) << "\n"
              << "session" << session_id
              << "_ram_after_reload_mb="
              << (report.ram_after_reload_bytes / (1024 * 1024)) << "\n"
              << "session" << session_id
              << "_prefill_ms=" << report.prefill_ms << "\n"
              << "session" << session_id
              << "_save_state_ms=" << report.save_state_ms << "\n"
              << "session" << session_id
              << "_evict_context_ms=" << report.evict_context_ms << "\n"
              << "session" << session_id
              << "_evict_model_ms=" << report.evict_model_ms << "\n"
              << "session" << session_id
              << "_restore_state_ms=" << report.restore_state_ms << "\n"
              << "session" << session_id
              << "_resume_check_ms=" << report.resume_check_ms << "\n"
              << "session" << session_id
              << "_restore_generate_ms=" << report.restore_generate_ms << "\n";
    if (!report.generated_text.empty() || report.generated_tokens > 0) {
      std::cout << "session" << session_id
                << "_generated_tokens=" << report.generated_tokens << "\n"
                << "session" << session_id
                << "_generated_contains_expected="
                << BoolText(report.generated_contains_expected) << "\n"
                << "session" << session_id
                << "_generated_text=" << EscapeReportValue(report.generated_text)
                << "\n";
    }
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
  if (session.backend == "onnx-llm") {
    const auto* adapter =
        dynamic_cast<const OnnxLlmResidencyAdapter*>(session.adapter.get());
    if (adapter == nullptr) {
      return;
    }
    const OnnxLlmResidencyReport& report = adapter->report();
    std::cout << "session" << session_id
              << "_prompt_tokens=" << report.prompt_tokens << "\n"
              << "session" << session_id
              << "_prefill_chunks=" << report.prefill_chunks << "\n"
              << "session" << session_id << "_prefill_chunk_tokens="
              << report.prefill_chunk_tokens << "\n"
              << "session" << session_id
              << "_logits_dtype=" << report.logits_dtype << "\n"
              << "session" << session_id
              << "_logits_vocab_size=" << report.logits_vocab_size << "\n"
              << "session" << session_id
              << "_logits_finite_count=" << report.logits_finite_count << "\n"
              << "session" << session_id
              << "_logits_nan_count=" << report.logits_nan_count << "\n"
              << "session" << session_id
              << "_logits_pos_inf_count=" << report.logits_pos_inf_count
              << "\n"
              << "session" << session_id
              << "_logits_neg_inf_count=" << report.logits_neg_inf_count
              << "\n"
              << "session" << session_id
              << "_decode_valid=" << BoolText(report.decode_valid) << "\n"
              << "session" << session_id << "_position_ids_present="
              << BoolText(report.position_ids_present) << "\n"
              << "session" << session_id
              << "_kv_state_bytes=" << report.kv_state_bytes << "\n"
              << "session" << session_id
              << "_restored_kv_state_bytes="
              << report.restored_kv_state_bytes << "\n"
              << "session" << session_id
              << "_baseline_next_token=" << report.baseline_next_token << "\n"
              << "session" << session_id
              << "_restored_next_token=" << report.restored_next_token << "\n"
              << "session" << session_id
              << "_resume_match=" << BoolText(report.resume_match) << "\n"
              << "session" << session_id << "_initial_session_load_ms="
              << report.initial_session_load_ms << "\n"
              << "session" << session_id
              << "_session_reload_ms=" << report.session_reload_ms << "\n"
              << "session" << session_id
              << "_prefill_ms=" << report.prefill_ms << "\n"
              << "session" << session_id
              << "_save_state_ms=" << report.save_state_ms << "\n"
              << "session" << session_id
              << "_evict_context_ms=" << report.evict_context_ms << "\n"
              << "session" << session_id
              << "_evict_model_ms=" << report.evict_model_ms << "\n"
              << "session" << session_id
              << "_restore_state_ms=" << report.restore_state_ms << "\n"
              << "session" << session_id
              << "_resume_check_ms=" << report.resume_check_ms << "\n"
              << "session" << session_id
              << "_restore_generate_ms=" << report.restore_generate_ms << "\n";
    if (!report.generated_token_ids.empty() || report.generated_tokens > 0) {
      std::cout << "session" << session_id
                << "_generated_tokens=" << report.generated_tokens << "\n"
                << "session" << session_id
                << "_generated_contains_expected="
                << BoolText(report.generated_contains_expected) << "\n"
                << "session" << session_id
                << "_generated_token_ids="
                << TokenListText(report.generated_token_ids) << "\n";
    }
    return;
  }
#endif  // MOSAICVRAM_ENABLE_ONNX
}

SessionRuntime* FindSession(std::map<MosaicSessionId, SessionRuntime>* sessions,
                            MosaicSessionId session_id) {
  const auto it = sessions->find(session_id);
  if (it == sessions->end()) {
    throw std::runtime_error("unknown session id in step");
  }
  return &it->second;
}

void CheckResult(const ResidencyControllerResult& result, int line_number) {
  if (!result.ok) {
    std::ostringstream message;
    message << "line " << line_number << " failed: " << result.error_message;
    throw std::runtime_error(message.str());
  }
}

int RunPlan(const Options& options) {
  const std::vector<PlanLine> lines = LoadPlan(options.plan_path);
  ResidencyController controller;
  std::map<MosaicSessionId, SessionRuntime> sessions;
  int executed_steps = 0;

  for (const PlanLine& line : lines) {
    if (line.kind != "session") {
      continue;
    }
    const MosaicSessionId session_id = RequiredSessionId(line, "id");
    if (sessions.find(session_id) != sessions.end()) {
      std::ostringstream message;
      message << "line " << line.line_number
              << " duplicates session id: " << session_id;
      throw std::runtime_error(message.str());
    }
    SessionRuntime session;
    session.backend = RequiredValue(line, "backend");
    session.adapter = CreateAdapter(line);
    BackendStateAdapter* adapter = session.adapter.get();
    sessions.emplace(session_id, std::move(session));
    CheckResult(controller.RegisterAdapter(adapter), line.line_number);
  }

  for (const PlanLine& line : lines) {
    if (line.kind == "session") {
      continue;
    }
    if (line.kind != "step") {
      std::ostringstream message;
      message << "line " << line.line_number
              << " has unknown directive: " << line.kind;
      throw std::runtime_error(message.str());
    }

    const std::string& op = RequiredValue(line, "op");
    const MosaicSessionId session_id = RequiredSessionId(line, "session");
    SessionRuntime* session = FindSession(&sessions, session_id);

    if (op == "load") {
      LoadSession(session);
    } else if (op == "prefill") {
      PrefillSession(session);
    } else if (op == "baseline") {
      CaptureBaseline(session);
    } else if (op == "save") {
      CheckResult(controller.SaveSession(session_id), line.line_number);
    } else if (op == "evict_context") {
      CheckResult(controller.EvictContext(session_id), line.line_number);
    } else if (op == "evict_model") {
      CheckResult(controller.EvictModel(session_id), line.line_number);
    } else if (op == "reload_model") {
      CheckResult(controller.ReloadModel(session_id), line.line_number);
    } else if (op == "restore") {
      CheckResult(controller.RestoreSession(session_id), line.line_number);
    } else if (op == "resume_check") {
      CheckResult(controller.ResumeCheck(session_id), line.line_number);
    } else if (op == "restore_then_generate") {
      RestoreThenGenerate(session, line);
    } else if (op == "switch_gpu_owner") {
      const MosaicSessionId to_session_id = RequiredSessionId(line, "to");
      CheckResult(controller.SwitchGpuOwner(session_id, to_session_id),
                  line.line_number);
    } else {
      std::ostringstream message;
      message << "line " << line.line_number << " has unknown op: " << op;
      throw std::runtime_error(message.str());
    }
    ++executed_steps;
  }

  std::cout << "command=residency-run\n"
            << "plan_path=" << options.plan_path << "\n"
            << "sessions=" << sessions.size() << "\n"
            << "steps=" << executed_steps << "\n"
            << "gpu_owner=" << controller.gpu_owner() << "\n";
  for (const auto& [session_id, session] : sessions) {
    std::cout << "session" << session_id << "_backend=" << session.backend
              << "\n"
              << "session" << session_id << "_residency="
              << ResidencyStateName(controller.ResidencyOf(session_id)) << "\n";
    PrintAdapterReport(session_id, session);
  }
  std::cout << "status=ok\n";
  return EXIT_SUCCESS;
}

}  // namespace

int RunResidencyRunCommand(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    return EXIT_FAILURE;
  }
  if (options.help_requested) {
    PrintUsage();
    return EXIT_SUCCESS;
  }
  try {
    return RunPlan(options);
  } catch (const std::exception& error) {
    std::cerr << "residency-run failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}

}  // namespace mosaicvram
