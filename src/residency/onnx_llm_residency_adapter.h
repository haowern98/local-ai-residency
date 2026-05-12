#ifndef MOSAICVRAM_SRC_RESIDENCY_ONNX_LLM_RESIDENCY_ADAPTER_H_
#define MOSAICVRAM_SRC_RESIDENCY_ONNX_LLM_RESIDENCY_ADAPTER_H_

#ifdef MOSAICVRAM_ENABLE_ONNX

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "residency/backend_state_adapter.h"

namespace mosaicvram {

struct OnnxLlmResidencyOptions {
  MosaicSessionId session_id = 2;
  std::string model_path;
  std::vector<int64_t> prompt_tokens;
  int prefill_chunk_tokens = 512;
  int device_index = 0;
  std::size_t max_snapshot_bytes = 0;
};

struct OnnxLlmResidencyReport {
  int past_input_count = 0;
  int present_output_count = 0;
  int logits_output_count = 0;
  std::string logits_dtype;
  std::size_t logits_vocab_size = 0;
  std::size_t logits_finite_count = 0;
  std::size_t logits_nan_count = 0;
  std::size_t logits_pos_inf_count = 0;
  std::size_t logits_neg_inf_count = 0;
  std::size_t prompt_tokens = 0;
  std::size_t prefill_chunks = 0;
  int prefill_chunk_tokens = 0;
  std::size_t snapshot_bytes = 0;
  std::size_t kv_state_bytes = 0;
  std::size_t restored_kv_state_bytes = 0;
  std::size_t generated_tokens = 0;
  int64_t first_token = -1;
  int64_t baseline_next_token = -1;
  int64_t restored_next_token = -1;
  double baseline_logits_checksum = 0.0;
  double restored_logits_checksum = 0.0;
  double initial_session_load_ms = 0.0;
  double session_reload_ms = 0.0;
  double prefill_ms = 0.0;
  double save_state_ms = 0.0;
  double evict_context_ms = 0.0;
  double evict_model_ms = 0.0;
  double restore_state_ms = 0.0;
  double resume_check_ms = 0.0;
  double restore_generate_ms = 0.0;
  std::size_t max_snapshot_bytes = 0;
  bool position_ids_present = false;
  bool decode_valid = false;
  bool cache_surface_found = false;
  bool resume_match = false;
  bool generated_contains_expected = false;
  std::vector<int64_t> generated_token_ids;
};

class OnnxLlmResidencyAdapter : public BackendStateAdapter {
 public:
  explicit OnnxLlmResidencyAdapter(OnnxLlmResidencyOptions options);
  ~OnnxLlmResidencyAdapter() override;

  OnnxLlmResidencyAdapter(const OnnxLlmResidencyAdapter&) = delete;
  OnnxLlmResidencyAdapter& operator=(const OnnxLlmResidencyAdapter&) = delete;

  void Load();
  void PrefillPrompt();
  void CaptureBaselineNextToken();
  void RestoreAndGenerateContinuation(
      const std::vector<int64_t>& tokens, int max_tokens,
      const std::vector<int64_t>& expected_tokens);

  BackendStateSnapshot& SaveState() override;
  void EvictContext() override;
  void EvictModel() override;
  void ReloadModel() override;
  void RestoreState() override;
  bool ResumeCheck() override;
  ResidencyState residency_state() const override { return residency_state_; }
  MosaicSessionId session_id() const override { return options_.session_id; }

  const OnnxLlmResidencyReport& report() const { return report_; }

 private:
  struct CudaBuffer {
    ~CudaBuffer();

    CudaBuffer() = default;
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    void Allocate(std::size_t bytes);
    void Free();

    void* data = nullptr;
    std::size_t bytes = 0;
  };

  struct KvTensor {
    std::string past_name;
    std::string present_name;
    ONNXTensorElementDataType element_type =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<int64_t> base_shape;
    std::vector<int64_t> cache_shape;
    CudaBuffer cache_buffer;
  };

  struct DecodeResult {
    int64_t next_token = -1;
    double logits_checksum = 0.0;
  };

  void CreateSession(double* elapsed_ms);
  void DiscoverModelIo();
  void AllocateInitialCache();
  DecodeResult RunDecodeStep(const std::vector<int64_t>& input_tokens,
                             int64_t past_length, bool read_logits);
  void ReplaceCache(std::vector<KvTensor>* output_tensors,
                    int64_t cache_length);
  void FreeCache();
  void ValidateReadyForDecode() const;
  std::size_t CacheBytes() const;

  OnnxLlmResidencyOptions options_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_ids_name_;
  std::string attention_mask_name_;
  std::string position_ids_name_;
  std::string logits_name_;
  ONNXTensorElementDataType logits_element_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  int64_t vocab_size_ = 0;
  int64_t cache_length_ = 0;
  int64_t snapshot_cache_length_ = 0;
  std::vector<KvTensor> kv_tensors_;
  BackendStateSnapshot snapshot_;
  OnnxLlmResidencyReport report_;
  ResidencyState residency_state_ = ResidencyState::kUnloaded;
};

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_ONNX

#endif  // MOSAICVRAM_SRC_RESIDENCY_ONNX_LLM_RESIDENCY_ADAPTER_H_
