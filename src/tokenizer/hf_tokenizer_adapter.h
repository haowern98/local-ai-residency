#ifndef MOSAICVRAM_SRC_TOKENIZER_HF_TOKENIZER_ADAPTER_H_
#define MOSAICVRAM_SRC_TOKENIZER_HF_TOKENIZER_ADAPTER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace mosaicvram {

struct HfTokenizerOptions {
  std::string tokenizer_path;
  std::string python_executable = "python";
};

std::vector<int64_t> TokenizeWithHfTokenizer(
    const HfTokenizerOptions& options, const std::string& prompt);

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_TOKENIZER_HF_TOKENIZER_ADAPTER_H_
