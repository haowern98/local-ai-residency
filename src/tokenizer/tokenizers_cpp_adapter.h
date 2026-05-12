#ifndef MOSAICVRAM_SRC_TOKENIZER_TOKENIZERS_CPP_ADAPTER_H_
#define MOSAICVRAM_SRC_TOKENIZER_TOKENIZERS_CPP_ADAPTER_H_

#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON

#include <cstdint>
#include <string>
#include <vector>

namespace mosaicvram {

std::vector<int64_t> TokenizeWithTokenizerJson(
    const std::string& tokenizer_path, const std::string& prompt);

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON

#endif  // MOSAICVRAM_SRC_TOKENIZER_TOKENIZERS_CPP_ADAPTER_H_
