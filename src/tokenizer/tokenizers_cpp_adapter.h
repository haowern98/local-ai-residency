#ifndef MOSAICVRAM_SRC_TOKENIZER_TOKENIZERS_CPP_ADAPTER_H_
#define MOSAICVRAM_SRC_TOKENIZER_TOKENIZERS_CPP_ADAPTER_H_

#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mosaicvram {

struct TokenizerChatPrompt {
  std::string text;
  std::vector<std::string> stop_strings;
};

struct TokenizerChatMessage {
  std::string role;
  std::string content;
};

std::vector<int64_t> TokenizeWithTokenizerJson(
    const std::string& tokenizer_path, const std::string& prompt);

std::string DecodeWithTokenizerJson(const std::string& tokenizer_path,
                                    const std::vector<int64_t>& token_ids);

std::optional<int64_t> TokenIdForTokenizerString(
    const std::string& tokenizer_path, const std::string& token);

TokenizerChatPrompt ApplyTokenizerChatTemplate(
    const std::string& tokenizer_path, const std::string& user_text);

TokenizerChatPrompt ApplyTokenizerChatTemplate(
    const std::string& tokenizer_path,
    const std::vector<TokenizerChatMessage>& messages, bool add_assistant);

TokenizerChatPrompt ApplyTokenizerChatTurnTemplate(
    const std::string& tokenizer_path, const std::string& user_text,
    bool first_turn);

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON

#endif  // MOSAICVRAM_SRC_TOKENIZER_TOKENIZERS_CPP_ADAPTER_H_
