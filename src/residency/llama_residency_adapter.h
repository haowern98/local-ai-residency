#ifndef MOSAICVRAM_SRC_RESIDENCY_LLAMA_RESIDENCY_ADAPTER_H_
#define MOSAICVRAM_SRC_RESIDENCY_LLAMA_RESIDENCY_ADAPTER_H_

#ifdef MOSAICVRAM_ENABLE_LLAMA

#include <ggml.h>
#include <llama.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "residency/backend_state_adapter.h"

namespace mosaicvram {

struct LlamaResidencyOptions {
  MosaicSessionId session_id = 1;
  std::string model_path;
  std::string prompt;
  int context_tokens = 256;
  int batch_tokens = 256;
  int gpu_layers = -1;
  int device_index = 0;
  int threads = 8;
  llama_seq_id sequence_id = 0;
  bool use_mmap = true;
  bool use_mlock = false;
  bool use_direct_io = false;
  bool check_tensors = false;
};

struct LlamaResidencyReport {
  std::size_t prompt_tokens = 0;
  std::size_t full_state_bytes = 0;
  std::size_t sequence_state_bytes = 0;
  std::size_t restored_full_bytes = 0;
  std::size_t restored_sequence_bytes = 0;
  std::size_t recreated_full_bytes = 0;
  std::size_t recreated_sequence_bytes = 0;
  std::size_t same_context_restore_bytes = 0;
  std::size_t model_reloaded_full_bytes = 0;
  std::size_t model_reloaded_sequence_bytes = 0;
  std::size_t generated_tokens = 0;
  llama_token first_token = LLAMA_TOKEN_NULL;
  llama_token baseline_next_token = LLAMA_TOKEN_NULL;
  llama_token full_restore_next_token = LLAMA_TOKEN_NULL;
  llama_token sequence_restore_next_token = LLAMA_TOKEN_NULL;
  llama_token recreated_full_next_token = LLAMA_TOKEN_NULL;
  llama_token recreated_sequence_next_token = LLAMA_TOKEN_NULL;
  llama_token same_context_restore_next_token = LLAMA_TOKEN_NULL;
  llama_token model_reloaded_full_next_token = LLAMA_TOKEN_NULL;
  llama_token model_reloaded_sequence_next_token = LLAMA_TOKEN_NULL;
  bool full_restore_match = false;
  bool sequence_restore_match = false;
  bool recreated_full_restore_match = false;
  bool recreated_sequence_restore_match = false;
  bool same_context_restore_match = false;
  bool same_context_pointer_preserved = false;
  bool model_reloaded_full_restore_match = false;
  bool model_reloaded_sequence_restore_match = false;
  bool generated_contains_expected = false;
  double initial_model_load_ms = 0.0;
  double model_reload_ms = 0.0;
  double context_reload_ms = 0.0;
  double prefill_ms = 0.0;
  double save_state_ms = 0.0;
  double evict_context_ms = 0.0;
  double evict_model_ms = 0.0;
  double restore_state_ms = 0.0;
  double resume_check_ms = 0.0;
  double restore_generate_ms = 0.0;
  std::string generated_text;
};

class LlamaResidencyAdapter : public BackendStateAdapter {
 public:
  explicit LlamaResidencyAdapter(LlamaResidencyOptions options);
  ~LlamaResidencyAdapter() override;

  LlamaResidencyAdapter(const LlamaResidencyAdapter&) = delete;
  LlamaResidencyAdapter& operator=(const LlamaResidencyAdapter&) = delete;

  void Load();
  void CreateContext();
  void PrefillPrompt();
  void CaptureBaselineNextToken();
  void CheckFullRestore();
  void CheckSequenceRestore();
  void CheckSameContextClearAndRestore();
  void RecreateContext();
  void CheckRecreatedFullRestore();
  void CheckRecreatedSequenceRestore();
  void CheckModelReloadedFullRestore();
  void CheckModelReloadedSequenceRestore();
  void RestoreAndGenerateContinuation(const std::string& text, int max_tokens,
                                      const std::string& expected_text);

  BackendStateSnapshot& SaveState() override;
  void EvictContext() override;
  void EvictModel() override;
  void ReloadModel() override;
  void RestoreState() override;
  bool ResumeCheck() override;
  ResidencyState residency_state() const override { return residency_state_; }
  MosaicSessionId session_id() const override { return options_.session_id; }

  const LlamaResidencyOptions& options() const { return options_; }
  const LlamaResidencyReport& report() const { return report_; }
  int context_tokens() const;

 private:
  class BackendLifetime {
   public:
    BackendLifetime();
    ~BackendLifetime();

    BackendLifetime(const BackendLifetime&) = delete;
    BackendLifetime& operator=(const BackendLifetime&) = delete;
  };

  struct ModelDeleter {
    void operator()(llama_model* model) const;
  };

  struct ContextDeleter {
    void operator()(llama_context* context) const;
  };

  using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
  using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;

  static void QuietLog(ggml_log_level level, const char* text, void* user_data);

  ModelPtr LoadModel() const;
  ContextPtr MakeContext() const;
  std::vector<llama_token> TokenizePrompt() const;
  std::vector<llama_token> TokenizeText(const std::string& text,
                                        bool add_special) const;
  void DecodeTokens(const std::vector<llama_token>& tokens);
  llama_token GreedyToken() const;
  std::string DetokenizeTokens(const std::vector<llama_token>& tokens) const;
  std::size_t RestoreFullState();
  std::size_t RestoreSequenceState();
  void ResetDecodePosition();

  LlamaResidencyOptions options_;
  BackendLifetime backend_lifetime_;
  ModelPtr model_;
  ContextPtr context_;
  const llama_vocab* vocab_ = nullptr;
  BackendStateSnapshot snapshot_;
  LlamaResidencyReport report_;
  ResidencyState residency_state_ = ResidencyState::kUnloaded;
  llama_pos next_decode_position_ = 0;
  llama_pos snapshot_decode_position_ = 0;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_LLAMA

#endif  // MOSAICVRAM_SRC_RESIDENCY_LLAMA_RESIDENCY_ADAPTER_H_
