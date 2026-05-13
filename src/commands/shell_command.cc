#include "commands/shell_command.h"

#include <charconv>
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

#include "residency/backend_state_adapter.h"
#include "residency/residency_controller.h"
#include "runtime/config_line.h"

#ifdef MOSAICVRAM_ENABLE_LLAMA
#include "residency/llama_residency_adapter.h"
#endif  // MOSAICVRAM_ENABLE_LLAMA

#ifdef MOSAICVRAM_ENABLE_ONNX
#include "residency/onnx_llm_residency_adapter.h"
#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON
#include "tokenizer/tokenizers_cpp_adapter.h"
#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON
#endif  // MOSAICVRAM_ENABLE_ONNX

namespace mosaicvram {
namespace {

struct ShellOptions {
  std::string config_path;
  bool help_requested = false;
};

struct SessionSpec {
  MosaicSessionId id = 0;
  std::string name;
  std::map<std::string, std::string> values;
  std::string issue;
};

struct ShellSession {
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

const std::string& RequiredValue(const SessionSpec& spec,
                                 const std::string& key) {
  const auto it = spec.values.find(key);
  if (it == spec.values.end() || it->second.empty()) {
    throw std::runtime_error("missing required key: " + key);
  }
  return it->second;
}

std::string OptionalString(const SessionSpec& spec, const std::string& key,
                           const std::string& default_value) {
  const auto it = spec.values.find(key);
  return it == spec.values.end() ? default_value : it->second;
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

void PrintUsage() {
  std::cout << "Usage: mosaicvram.exe shell [--config <path>]\n";
}

bool ParseOptions(int argc, char** argv, ShellOptions* options) {
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
    std::cerr << "Unknown shell option: " << arg << "\n";
    return false;
  }
  return true;
}

void ValidateSessionSpec(SessionSpec* spec) {
  spec->issue.clear();
  const auto backend = spec->values.find("backend");
  if (backend == spec->values.end() || backend->second.empty()) {
    spec->issue = "missing required key: backend";
    return;
  }
  if (backend->second == "llama") {
    const auto model = spec->values.find("model");
    if (model == spec->values.end() || model->second.empty()) {
      spec->issue = "missing required key: model";
    }
    return;
  }
  if (backend->second == "onnx-llm") {
    const auto model = spec->values.find("model");
    if (model == spec->values.end() || model->second.empty()) {
      spec->issue = "missing required key: model";
      return;
    }
    const auto tokenizer = spec->values.find("tokenizer");
    if (tokenizer == spec->values.end() || tokenizer->second.empty()) {
      spec->issue = "missing required key: tokenizer";
    }
    return;
  }
  spec->issue = "unsupported backend: " + backend->second;
}

class ShellRuntime {
 public:
  void LoadConfig(const std::string& path);
  void SaveConfig(const std::string& path) const;
  void AddSession(const std::string& name,
                  const std::map<std::string, std::string>& values);
  void SetValue(const std::string& name, const std::string& key,
                const std::string& value);
  void RemoveSession(const std::string& name);
  void UseSession(const std::string& name);
  void PrintSessions() const;
  void PrintStatus() const;
  void PrintSession(const std::string& name) const;
  void Load(const std::string& name);
  void Prefill(const std::string& name, const std::string& text);
  void Chat(const std::string& name, const std::string& text, int max_tokens);
  void Save(const std::string& name);
  void Evict(const std::string& name, const std::string& target);
  void Reload(const std::string& name);
  void Restore(const std::string& name);
  void ResumeCheck(const std::string& name);
  std::string ResolveName(const std::vector<std::string>& positional) const;
  bool HasSession(const std::string& name) const;
  bool empty() const { return sessions_.empty(); }

 private:
  ShellSession* FindMutable(const std::string& name);
  const ShellSession* Find(const std::string& name) const;
  void EnsureValid(const ShellSession& session) const;
  void DetachEvictedAdapter(ShellSession* session);
  void EnsureChatReady(const std::string& name, ShellSession* session);
  void RebuildAdapter(ShellSession* session, const std::string& prompt);

  std::map<std::string, ShellSession> sessions_;
  std::string active_session_;
  MosaicSessionId next_session_id_ = 1;
  ResidencyController controller_;
};

void ShellRuntime::LoadConfig(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    std::cout << "config not found: " << path << "\n";
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
      std::cout << "ignored config line " << line_number
                << ": expected session <name>\n";
      continue;
    }
    AddSession(line.positional.front(), line.values);
  }
}

void ShellRuntime::SaveConfig(const std::string& path) const {
  if (path.empty()) {
    throw std::runtime_error("save-config requires a path");
  }
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to open config for writing: " + path);
  }
  for (const auto& [name, session] : sessions_) {
    output << "session " << name;
    for (const auto& [key, value] : session.spec.values) {
      output << " " << key << "=\"" << value << "\"";
    }
    output << "\n";
  }
  std::cout << "saved " << path << "\n";
}

void ShellRuntime::AddSession(
    const std::string& name, const std::map<std::string, std::string>& values) {
  if (name.empty()) {
    throw std::runtime_error("session name is required");
  }
  if (sessions_.find(name) != sessions_.end()) {
    throw std::runtime_error("session already exists: " + name);
  }
  ShellSession session;
  session.spec.id = next_session_id_++;
  session.spec.name = name;
  session.spec.values = values;
  ValidateSessionSpec(&session.spec);
  sessions_.emplace(name, std::move(session));
  if (active_session_.empty()) {
    active_session_ = name;
  }
  std::cout << "added session " << name << "\n";
}

void ShellRuntime::SetValue(const std::string& name, const std::string& key,
                            const std::string& value) {
  ShellSession* session = FindMutable(name);
  if (session->adapter != nullptr &&
      (key == "model" || key == "tokenizer" || key == "backend")) {
    if (session->adapter->residency_state() != ResidencyState::kModelEvicted) {
      throw std::runtime_error("evict model before changing " + key);
    }
    DetachEvictedAdapter(session);
  }
  session->spec.values[key] = value;
  ValidateSessionSpec(&session->spec);
  std::cout << "set " << name << " " << key << "\n";
}

void ShellRuntime::RemoveSession(const std::string& name) {
  ShellSession* session = FindMutable(name);
  if (session->adapter != nullptr) {
    if (session->adapter->residency_state() != ResidencyState::kModelEvicted) {
      throw std::runtime_error("evict model before removing session");
    }
    DetachEvictedAdapter(session);
  }
  sessions_.erase(name);
  if (active_session_ == name) {
    active_session_ = sessions_.empty() ? "" : sessions_.begin()->first;
  }
}

void ShellRuntime::UseSession(const std::string& name) {
  (void)Find(name);
  active_session_ = name;
  std::cout << "active session: " << name << "\n";
}

void ShellRuntime::PrintSessions() const {
  if (sessions_.empty()) {
    std::cout << "No sessions configured.\n";
    return;
  }
  std::cout << "session backend residency issue\n";
  for (const auto& [name, session] : sessions_) {
    const std::string backend = OptionalString(session.spec, "backend", "-");
    const std::string residency =
        session.adapter == nullptr
            ? "Unloaded"
            : ResidencyStateName(session.adapter->residency_state());
    std::cout << name << " " << backend << " " << residency << " "
              << (session.spec.issue.empty() ? "-" : session.spec.issue)
              << "\n";
  }
}

void ShellRuntime::PrintStatus() const {
  std::cout << "active: " << (active_session_.empty() ? "-" : active_session_)
            << "\n"
            << "gpu_owner: " << controller_.gpu_owner() << "\n";
  PrintSessions();
}

void ShellRuntime::PrintSession(const std::string& name) const {
  const ShellSession* session = Find(name);
  std::cout << "session " << name << "\n";
  for (const auto& [key, value] : session->spec.values) {
    std::cout << "  " << key << "=" << value << "\n";
  }
  if (!session->spec.issue.empty()) {
    std::cout << "  issue=" << session->spec.issue << "\n";
  }
}

void ShellRuntime::Load(const std::string& name) {
  ShellSession* session = FindMutable(name);
  EnsureValid(*session);
  if (session->adapter == nullptr) {
    RebuildAdapter(session, "");
    const ResidencyControllerResult result =
        controller_.RegisterAdapter(session->adapter.get());
    if (!result.ok) {
      throw std::runtime_error(result.error_message);
    }
  }
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (session->spec.values["backend"] == "llama") {
    auto* adapter =
        dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get());
    adapter->Load();
    adapter->CreateContext();
    std::cout << "loaded " << name << "\n";
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA
#ifdef MOSAICVRAM_ENABLE_ONNX
  if (session->spec.values["backend"] == "onnx-llm") {
    auto* adapter =
        dynamic_cast<OnnxLlmResidencyAdapter*>(session->adapter.get());
    adapter->Load();
    std::cout << "loaded " << name << "\n";
    return;
  }
#endif  // MOSAICVRAM_ENABLE_ONNX
  throw std::runtime_error("unsupported backend for load");
}

void ShellRuntime::Prefill(const std::string& name, const std::string& text) {
  ShellSession* session = FindMutable(name);
  EnsureValid(*session);
  if (session->adapter != nullptr) {
    throw std::runtime_error("prefill requires an unloaded session");
  }
  RebuildAdapter(session, text);
  const ResidencyControllerResult result =
      controller_.RegisterAdapter(session->adapter.get());
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  Load(name);
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (session->spec.values["backend"] == "llama") {
    dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get())
        ->PrefillPrompt();
    std::cout << "prefilled " << name << "\n";
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA
#ifdef MOSAICVRAM_ENABLE_ONNX
  if (session->spec.values["backend"] == "onnx-llm") {
    dynamic_cast<OnnxLlmResidencyAdapter*>(session->adapter.get())
        ->PrefillPrompt();
    std::cout << "prefilled " << name << "\n";
    return;
  }
#endif  // MOSAICVRAM_ENABLE_ONNX
}

void ShellRuntime::Chat(const std::string& name, const std::string& text,
                        int max_tokens) {
  ShellSession* session = FindMutable(name);
  EnsureValid(*session);
  EnsureChatReady(name, session);
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (session->spec.values["backend"] == "llama") {
    const std::string generated =
        dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get())
            ->GenerateContinuation(text, max_tokens);
    std::cout << name << ": " << generated << "\n";
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA
#ifdef MOSAICVRAM_ENABLE_ONNX
  if (session->spec.values["backend"] == "onnx-llm") {
#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON
    const std::string tokenizer = RequiredValue(session->spec, "tokenizer");
    const std::vector<int64_t> tokens =
        TokenizeWithTokenizerJson(tokenizer, text);
    const std::vector<int64_t> generated =
        dynamic_cast<OnnxLlmResidencyAdapter*>(session->adapter.get())
            ->GenerateContinuation(tokens, max_tokens);
    std::cout << name << " generated token_ids=" << TokenListText(generated)
              << "\n";
    return;
#else
    throw std::runtime_error(
        "ONNX chat requires tokenizer.json support in this build");
#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON
  }
#endif  // MOSAICVRAM_ENABLE_ONNX
  throw std::runtime_error("unsupported backend for chat");
}

void ShellRuntime::Save(const std::string& name) {
  ShellSession* session = FindMutable(name);
  (void)session;
  const ResidencyControllerResult result =
      controller_.SaveSession(Find(name)->spec.id);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  std::cout << "saved " << name << "\n";
}

void ShellRuntime::Evict(const std::string& name, const std::string& target) {
  const MosaicSessionId id = Find(name)->spec.id;
  ResidencyControllerResult result;
  if (target == "context") {
    result = controller_.EvictContext(id);
  } else if (target == "model") {
    result = controller_.EvictModel(id);
  } else {
    throw std::runtime_error("evict target must be context or model");
  }
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  std::cout << "evicted " << name << " " << target << "\n";
}

void ShellRuntime::Reload(const std::string& name) {
  const ResidencyControllerResult result =
      controller_.ReloadModel(Find(name)->spec.id);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  std::cout << "reloaded " << name << "\n";
}

void ShellRuntime::Restore(const std::string& name) {
  const ResidencyControllerResult result =
      controller_.RestoreSession(Find(name)->spec.id);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  std::cout << "restored " << name << "\n";
}

void ShellRuntime::ResumeCheck(const std::string& name) {
  const ResidencyControllerResult result =
      controller_.ResumeCheck(Find(name)->spec.id);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  std::cout << "resume-check ok\n";
}

std::string ShellRuntime::ResolveName(
    const std::vector<std::string>& positional) const {
  if (!positional.empty()) {
    return positional.front();
  }
  if (active_session_.empty()) {
    throw std::runtime_error("no active session; use <session>");
  }
  return active_session_;
}

bool ShellRuntime::HasSession(const std::string& name) const {
  return sessions_.find(name) != sessions_.end();
}

ShellSession* ShellRuntime::FindMutable(const std::string& name) {
  auto it = sessions_.find(name);
  if (it == sessions_.end()) {
    throw std::runtime_error("unknown session: " + name);
  }
  return &it->second;
}

const ShellSession* ShellRuntime::Find(const std::string& name) const {
  auto it = sessions_.find(name);
  if (it == sessions_.end()) {
    throw std::runtime_error("unknown session: " + name);
  }
  return &it->second;
}

void ShellRuntime::EnsureValid(const ShellSession& session) const {
  if (!session.spec.issue.empty()) {
    throw std::runtime_error("invalid session " + session.spec.name + ": " +
                             session.spec.issue);
  }
}

void ShellRuntime::DetachEvictedAdapter(ShellSession* session) {
  const ResidencyControllerResult result =
      controller_.UnregisterAdapter(session->spec.id);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  session->adapter.reset();
}

void ShellRuntime::EnsureChatReady(const std::string& name,
                                   ShellSession* session) {
  if (session->adapter == nullptr ||
      session->adapter->residency_state() == ResidencyState::kUnloaded) {
    Load(name);
    return;
  }

  const ResidencyState state = session->adapter->residency_state();
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (session->spec.values["backend"] == "llama") {
    auto* adapter =
        dynamic_cast<LlamaResidencyAdapter*>(session->adapter.get());
    if (state == ResidencyState::kModelEvicted) {
      adapter->ReloadModel();
      adapter->CreateContext();
      std::cout << "reloaded " << name << "\n";
      return;
    }
    if (state == ResidencyState::kContextEvicted) {
      adapter->CreateContext();
      std::cout << "recreated context " << name << "\n";
      return;
    }
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA

  if (state == ResidencyState::kModelEvicted ||
      state == ResidencyState::kContextEvicted) {
    throw std::runtime_error(
        "session is evicted; use reload and restore first");
  }
}

void ShellRuntime::RebuildAdapter(ShellSession* session,
                                  const std::string& prompt) {
  const std::string& backend = RequiredValue(session->spec, "backend");
#ifdef MOSAICVRAM_ENABLE_LLAMA
  if (backend == "llama") {
    LlamaResidencyOptions options;
    options.session_id = session->spec.id;
    options.model_path = RequiredValue(session->spec, "model");
    options.prompt = prompt;
    options.context_tokens =
        OptionalInt(session->spec, "ctx", options.context_tokens);
    options.batch_tokens =
        OptionalInt(session->spec, "batch", options.batch_tokens);
    options.gpu_layers =
        OptionalInt(session->spec, "gpu_layers", options.gpu_layers);
    options.device_index =
        OptionalInt(session->spec, "device", options.device_index);
    options.threads = OptionalInt(session->spec, "threads", options.threads);
    options.sequence_id =
        OptionalInt(session->spec, "seq_id", options.sequence_id);
    session->adapter = std::make_unique<LlamaResidencyAdapter>(options);
    return;
  }
#endif  // MOSAICVRAM_ENABLE_LLAMA
#ifdef MOSAICVRAM_ENABLE_ONNX
  if (backend == "onnx-llm") {
#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON
    OnnxLlmResidencyOptions options;
    options.session_id = session->spec.id;
    options.model_path = RequiredValue(session->spec, "model");
    if (!prompt.empty()) {
      options.prompt_tokens = TokenizeWithTokenizerJson(
          RequiredValue(session->spec, "tokenizer"), prompt);
    }
    options.prefill_chunk_tokens = OptionalInt(session->spec, "prefill_chunk",
                                               options.prefill_chunk_tokens);
    options.device_index =
        OptionalInt(session->spec, "device", options.device_index);
    session->adapter = std::make_unique<OnnxLlmResidencyAdapter>(options);
    return;
#else
    throw std::runtime_error(
        "ONNX sessions require tokenizer.json support in this build");
#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON
  }
#endif  // MOSAICVRAM_ENABLE_ONNX
  throw std::runtime_error("unsupported backend: " + backend);
}

void PrintEmptyConfigHelp() {
  std::cout
      << "No sessions configured.\n\n"
      << "Add one:\n"
      << "  add llama backend=llama model=\"C:\\models\\qwen.gguf\" "
         "ctx=16384 batch=512 gpu_layers=-1 device=0\n"
      << "  add onnx backend=onnx-llm "
         "model=\"C:\\models\\onnx-model\\model.onnx\" "
         "tokenizer=\"C:\\models\\onnx-model\" prefill_chunk=512 device=0\n";
}

void PrintHelp() {
  std::cout << "Commands:\n"
            << "  sessions\n"
            << "  status\n"
            << "  show <session>\n"
            << "  add <session> backend=... model=...\n"
            << "  set <session> <key> <value>\n"
            << "  remove <session>\n"
            << "  use <session>\n"
            << "  load [session]\n"
            << "  prefill [session] \"text\"\n"
            << "  chat [session] \"text\" [max_tokens=N]\n"
            << "  save [session]\n"
            << "  evict [session] context|model\n"
            << "  reload [session]\n"
            << "  restore [session]\n"
            << "  resume-check [session]\n"
            << "  save-config [path]\n"
            << "  load-config <path>\n"
            << "  exit\n";
}

int OptionalMaxTokens(const ConfigLine& line) {
  const auto it = line.values.find("max_tokens");
  if (it == line.values.end()) {
    return 32;
  }
  int parsed = 0;
  if (!ParseInt(it->second, &parsed) || parsed <= 0) {
    throw std::runtime_error("max_tokens must be positive");
  }
  return parsed;
}

std::string TextArgument(const ConfigLine& line, std::size_t index) {
  if (line.positional.size() <= index) {
    throw std::runtime_error("missing text argument");
  }
  return line.positional[index];
}

std::string ResolveOptionalSessionText(const ConfigLine& line,
                                       const ShellRuntime& runtime,
                                       std::string* text) {
  if (line.positional.empty()) {
    throw std::runtime_error("missing text argument");
  }
  if (line.positional.size() >= 2 || runtime.HasSession(line.positional[0])) {
    *text = TextArgument(line, 1);
    return runtime.ResolveName({line.positional[0]});
  }
  *text = line.positional[0];
  return runtime.ResolveName({});
}

std::string ResolveOptionalSessionTarget(const ConfigLine& line,
                                         const ShellRuntime& runtime,
                                         const std::string& command_name) {
  if (line.positional.empty()) {
    throw std::runtime_error(command_name + " requires context or model");
  }
  if (line.positional.size() >= 2 || runtime.HasSession(line.positional[0])) {
    return runtime.ResolveName({line.positional[0]});
  }
  return runtime.ResolveName({});
}

void ExecuteLine(const ConfigLine& line, ShellRuntime* runtime,
                 std::string* config_path, bool* exit_requested) {
  if (line.kind.empty()) {
    return;
  }
  if (line.kind == "help") {
    PrintHelp();
    return;
  }
  if (line.kind == "exit" || line.kind == "quit") {
    *exit_requested = true;
    return;
  }
  if (line.kind == "sessions") {
    runtime->PrintSessions();
    return;
  }
  if (line.kind == "status") {
    runtime->PrintStatus();
    return;
  }
  if (line.kind == "show") {
    runtime->PrintSession(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "add") {
    if (line.positional.empty()) {
      throw std::runtime_error("add requires a session name");
    }
    runtime->AddSession(line.positional.front(), line.values);
    return;
  }
  if (line.kind == "set") {
    if (line.positional.size() < 3) {
      throw std::runtime_error("set requires: set <session> <key> <value>");
    }
    runtime->SetValue(line.positional[0], line.positional[1],
                      line.positional[2]);
    return;
  }
  if (line.kind == "remove") {
    runtime->RemoveSession(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "use") {
    runtime->UseSession(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "load") {
    runtime->Load(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "prefill") {
    std::string text;
    const std::string name = ResolveOptionalSessionText(line, *runtime, &text);
    runtime->Prefill(name, text);
    return;
  }
  if (line.kind == "chat") {
    std::string text;
    const std::string name = ResolveOptionalSessionText(line, *runtime, &text);
    runtime->Chat(name, text, OptionalMaxTokens(line));
    return;
  }
  if (line.kind == "save") {
    runtime->Save(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "evict") {
    const std::string name =
        ResolveOptionalSessionTarget(line, *runtime, "evict");
    const std::size_t target_index =
        runtime->HasSession(line.positional[0]) ? 1 : 0;
    if (line.positional.size() <= target_index) {
      throw std::runtime_error("evict requires context or model");
    }
    runtime->Evict(name, line.positional[target_index]);
    return;
  }
  if (line.kind == "reload") {
    runtime->Reload(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "restore") {
    runtime->Restore(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "resume-check") {
    runtime->ResumeCheck(runtime->ResolveName(line.positional));
    return;
  }
  if (line.kind == "save-config") {
    const std::string path =
        line.positional.empty() ? *config_path : line.positional.front();
    runtime->SaveConfig(path);
    if (!path.empty()) {
      *config_path = path;
    }
    return;
  }
  if (line.kind == "load-config") {
    if (line.positional.empty()) {
      throw std::runtime_error("load-config requires a path");
    }
    runtime->LoadConfig(line.positional.front());
    *config_path = line.positional.front();
    return;
  }
  throw std::runtime_error("unknown command: " + line.kind);
}

}  // namespace

int RunShellCommand(int argc, char** argv) {
  ShellOptions options;
  if (!ParseOptions(argc, argv, &options)) {
    return EXIT_FAILURE;
  }
  if (options.help_requested) {
    PrintUsage();
    return EXIT_SUCCESS;
  }

  ShellRuntime runtime;
  if (!options.config_path.empty()) {
    runtime.LoadConfig(options.config_path);
  }
  std::cout << "MosaicVRAM shell\n";
  if (runtime.empty()) {
    PrintEmptyConfigHelp();
  }

  bool exit_requested = false;
  std::string line_text;
  while (!exit_requested) {
    std::cout << "mosaic> ";
    if (!std::getline(std::cin, line_text)) {
      break;
    }
    try {
      ExecuteLine(ParseConfigLine(Trim(line_text), 0), &runtime,
                  &options.config_path, &exit_requested);
    } catch (const std::exception& error) {
      std::cout << "error: " << error.what() << "\n";
    }
  }
  return EXIT_SUCCESS;
}

}  // namespace mosaicvram
