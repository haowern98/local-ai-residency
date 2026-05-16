#include "tokenizer/tokenizers_cpp_adapter.h"

#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON

#include <tokenizers_cpp.h>

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mosaicvram {
namespace {

std::string ReadBinaryFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open tokenizer file: " + path.string());
  }
  std::ostringstream bytes;
  bytes << input.rdbuf();
  return bytes.str();
}

std::filesystem::path ResolveTokenizerJsonPath(const std::string& path) {
  const std::filesystem::path tokenizer_path(path);
  if (std::filesystem::is_regular_file(tokenizer_path)) {
    if (tokenizer_path.filename() != "tokenizer.json") {
      throw std::runtime_error(
          "tokenizer path must point to tokenizer.json for this adapter");
    }
    return tokenizer_path;
  }

  const std::filesystem::path json_path = tokenizer_path / "tokenizer.json";
  if (!std::filesystem::is_regular_file(json_path)) {
    throw std::runtime_error("tokenizer.json was not found under: " + path);
  }
  return json_path;
}

std::filesystem::path ResolveTokenizerConfigPath(const std::string& path) {
  const std::filesystem::path json_path = ResolveTokenizerJsonPath(path);
  return json_path.parent_path() / "tokenizer_config.json";
}

void AppendUtf8(std::uint32_t value, std::string* output) {
  if (value <= 0x7f) {
    output->push_back(static_cast<char>(value));
  } else if (value <= 0x7ff) {
    output->push_back(static_cast<char>(0xc0 | (value >> 6)));
    output->push_back(static_cast<char>(0x80 | (value & 0x3f)));
  } else if (value <= 0xffff) {
    output->push_back(static_cast<char>(0xe0 | (value >> 12)));
    output->push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | (value & 0x3f)));
  } else {
    output->push_back(static_cast<char>(0xf0 | (value >> 18)));
    output->push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | (value & 0x3f)));
  }
}

int HexValue(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  return -1;
}

std::uint32_t ParseJsonHex4(std::string_view text, std::size_t offset) {
  if (offset + 4 > text.size()) {
    throw std::runtime_error("invalid unicode escape in tokenizer_config.json");
  }
  std::uint32_t value = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    const int digit = HexValue(text[offset + i]);
    if (digit < 0) {
      throw std::runtime_error(
          "invalid unicode escape in tokenizer_config.json");
    }
    value = (value << 4) | static_cast<std::uint32_t>(digit);
  }
  return value;
}

std::string ParseJsonStringAt(std::string_view text, std::size_t quote_pos) {
  if (quote_pos >= text.size() || text[quote_pos] != '"') {
    throw std::runtime_error("expected JSON string");
  }

  std::string output;
  for (std::size_t i = quote_pos + 1; i < text.size(); ++i) {
    const char current = text[i];
    if (current == '"') {
      return output;
    }
    if (current != '\\') {
      output.push_back(current);
      continue;
    }
    if (++i >= text.size()) {
      throw std::runtime_error("unterminated JSON escape");
    }
    switch (text[i]) {
      case '"':
      case '\\':
      case '/':
        output.push_back(text[i]);
        break;
      case 'b':
        output.push_back('\b');
        break;
      case 'f':
        output.push_back('\f');
        break;
      case 'n':
        output.push_back('\n');
        break;
      case 'r':
        output.push_back('\r');
        break;
      case 't':
        output.push_back('\t');
        break;
      case 'u': {
        std::uint32_t value = ParseJsonHex4(text, i + 1);
        i += 4;
        if (value >= 0xd800 && value <= 0xdbff) {
          if (i + 6 > text.size() || text[i + 1] != '\\' ||
              text[i + 2] != 'u') {
            throw std::runtime_error(
                "invalid unicode surrogate in tokenizer_config.json");
          }
          const std::uint32_t low = ParseJsonHex4(text, i + 3);
          if (low < 0xdc00 || low > 0xdfff) {
            throw std::runtime_error(
                "invalid unicode surrogate in tokenizer_config.json");
          }
          value = 0x10000 + ((value - 0xd800) << 10) + (low - 0xdc00);
          i += 6;
        }
        AppendUtf8(value, &output);
        break;
      }
      default:
        throw std::runtime_error("unsupported JSON escape");
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

std::optional<std::string> ExtractJsonStringField(std::string_view json,
                                                  std::string_view field) {
  std::size_t search_pos = 0;
  while (search_pos < json.size()) {
    const std::size_t quote_pos = json.find('"', search_pos);
    if (quote_pos == std::string_view::npos) {
      return std::nullopt;
    }
    const std::string key = ParseJsonStringAt(json, quote_pos);
    std::size_t after_key = quote_pos + 1;
    bool escaped = false;
    while (after_key < json.size()) {
      const char current = json[after_key++];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (current == '\\') {
        escaped = true;
        continue;
      }
      if (current == '"') {
        break;
      }
    }
    if (key != field) {
      search_pos = after_key;
      continue;
    }
    const std::size_t colon = json.find(':', after_key);
    if (colon == std::string_view::npos) {
      throw std::runtime_error("invalid tokenizer_config.json");
    }
    const std::size_t value_quote = json.find('"', colon + 1);
    if (value_quote == std::string_view::npos) {
      throw std::runtime_error("invalid tokenizer_config.json");
    }
    return ParseJsonStringAt(json, value_quote);
  }
  return std::nullopt;
}

std::size_t SkipJsonString(std::string_view text, std::size_t quote_pos) {
  if (quote_pos >= text.size() || text[quote_pos] != '"') {
    throw std::runtime_error("expected JSON string");
  }
  bool escaped = false;
  for (std::size_t i = quote_pos + 1; i < text.size(); ++i) {
    const char current = text[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (current == '\\') {
      escaped = true;
      continue;
    }
    if (current == '"') {
      return i + 1;
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

bool IsDecimalInteger(std::string_view text) {
  if (text.empty()) {
    return false;
  }
  for (const char c : text) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  return true;
}

std::size_t FirstNonSpace(std::string_view text, std::size_t offset) {
  while (offset < text.size() &&
         std::isspace(static_cast<unsigned char>(text[offset]))) {
    ++offset;
  }
  return offset;
}

std::optional<int64_t> AddedTokenIdFromConfig(std::string_view config,
                                              std::string_view token) {
  std::size_t search_pos = 0;
  while (search_pos < config.size()) {
    const std::size_t quote_pos = config.find('"', search_pos);
    if (quote_pos == std::string_view::npos) {
      return std::nullopt;
    }
    const std::string key = ParseJsonStringAt(config, quote_pos);
    const std::size_t after_key = SkipJsonString(config, quote_pos);
    search_pos = after_key;
    if (!IsDecimalInteger(key)) {
      continue;
    }

    const std::size_t colon = config.find(':', after_key);
    if (colon == std::string_view::npos) {
      return std::nullopt;
    }
    const std::size_t object_open = FirstNonSpace(config, colon + 1);
    if (object_open >= config.size() || config[object_open] != '{') {
      continue;
    }

    const std::size_t next_key = config.find("\n    \"", object_open + 1);
    const std::size_t object_end =
        next_key == std::string_view::npos ? config.size() : next_key;
    const std::size_t content_key = config.find("\"content\"", object_open + 1);
    if (content_key == std::string_view::npos || content_key >= object_end) {
      continue;
    }
    const std::size_t content_colon = config.find(':', content_key);
    if (content_colon == std::string_view::npos ||
        content_colon >= object_end) {
      continue;
    }
    const std::size_t value_quote = config.find('"', content_colon + 1);
    if (value_quote == std::string_view::npos || value_quote >= object_end) {
      continue;
    }
    if (ParseJsonStringAt(config, value_quote) == token) {
      return std::stoll(key);
    }
  }
  return std::nullopt;
}

std::optional<std::string_view> AddedTokenObjectFromConfig(
    std::string_view config, int64_t token_id) {
  const std::string key = "\"" + std::to_string(token_id) + "\"";
  const std::size_t key_pos = config.find(key);
  if (key_pos == std::string_view::npos) {
    return std::nullopt;
  }
  const std::size_t colon = config.find(':', key_pos + key.size());
  if (colon == std::string_view::npos) {
    return std::nullopt;
  }
  const std::size_t object_open = FirstNonSpace(config, colon + 1);
  if (object_open >= config.size() || config[object_open] != '{') {
    return std::nullopt;
  }
  const std::size_t next_key = config.find("\n    \"", object_open + 1);
  const std::size_t object_end =
      next_key == std::string_view::npos ? config.size() : next_key;
  return config.substr(object_open, object_end - object_open);
}

bool AddedTokenIsSpecial(std::string_view config, int64_t token_id) {
  const std::optional<std::string_view> object =
      AddedTokenObjectFromConfig(config, token_id);
  if (!object.has_value()) {
    return false;
  }
  const std::size_t special_key = object->find("\"special\"");
  if (special_key == std::string_view::npos) {
    return false;
  }
  const std::size_t colon = object->find(':', special_key);
  if (colon == std::string_view::npos) {
    return false;
  }
  const std::size_t value = FirstNonSpace(*object, colon + 1);
  return object->substr(value, 4) == "true";
}

std::unique_ptr<tokenizers::Tokenizer> LoadTokenizerJson(
    const std::string& tokenizer_path) {
  const std::filesystem::path json_path =
      ResolveTokenizerJsonPath(tokenizer_path);
  const std::string json_blob = ReadBinaryFile(json_path);
  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      tokenizers::Tokenizer::FromBlobJSON(json_blob);
  if (tokenizer == nullptr) {
    throw std::runtime_error("failed to load tokenizer.json: " +
                             json_path.string());
  }
  return tokenizer;
}

std::string ReadTokenizerConfig(const std::string& tokenizer_path) {
  const std::filesystem::path config_path =
      ResolveTokenizerConfigPath(tokenizer_path);
  if (!std::filesystem::is_regular_file(config_path)) {
    throw std::runtime_error("tokenizer_config.json was not found under: " +
                             config_path.parent_path().string());
  }
  return ReadBinaryFile(config_path);
}

std::string TokenizerConfigString(const std::string& tokenizer_path,
                                  const std::string& field,
                                  const std::string& default_value) {
  const std::string config = ReadTokenizerConfig(tokenizer_path);
  const std::optional<std::string> value =
      ExtractJsonStringField(config, field);
  return value.has_value() ? *value : default_value;
}

std::string RequiredTokenizerConfigString(const std::string& tokenizer_path,
                                          const std::string& field) {
  const std::string config = ReadTokenizerConfig(tokenizer_path);
  const std::optional<std::string> value =
      ExtractJsonStringField(config, field);
  if (!value.has_value() || value->empty()) {
    throw std::runtime_error("tokenizer_config.json missing required field: " +
                             field);
  }
  return *value;
}

bool Contains(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

std::optional<std::string> ExtractSingleQuotedLiteralContaining(
    std::string_view text, std::string_view needle) {
  std::optional<std::string> best;
  std::size_t search_pos = 0;
  while (search_pos < text.size()) {
    const std::size_t quote_pos = text.find('\'', search_pos);
    if (quote_pos == std::string_view::npos) {
      return best;
    }

    std::string literal;
    bool escaped = false;
    for (std::size_t i = quote_pos + 1; i < text.size(); ++i) {
      const char current = text[i];
      if (escaped) {
        switch (current) {
          case '\\':
          case '\'':
            literal.push_back(current);
            break;
          case 'n':
            literal.push_back('\n');
            break;
          default:
            literal.push_back(current);
            break;
        }
        escaped = false;
        continue;
      }
      if (current == '\\') {
        escaped = true;
        continue;
      }
      if (current == '\'') {
        if (Contains(literal, needle) &&
            (!best.has_value() || literal.size() < best->size())) {
          best = literal;
        }
        search_pos = i + 1;
        break;
      }
      literal.push_back(current);
    }
    if (escaped) {
      return std::nullopt;
    }
  }
  return best;
}

void AppendChatMessage(const TokenizerChatMessage& message,
                       const std::string& user_prefix,
                       const std::string& assistant_prefix,
                       const std::string& suffix, std::string* output) {
  if (message.role == "user") {
    *output += user_prefix;
  } else if (message.role == "assistant") {
    *output += assistant_prefix;
  } else {
    throw std::runtime_error("unsupported chat message role: " + message.role);
  }
  *output += message.content;
  *output += suffix;
}

}  // namespace

std::vector<int64_t> TokenizeWithTokenizerJson(
    const std::string& tokenizer_path, const std::string& prompt) {
  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      LoadTokenizerJson(tokenizer_path);

  const std::vector<int32_t> token_ids = tokenizer->Encode(prompt);
  if (token_ids.empty()) {
    throw std::runtime_error("tokenizer.json produced no token ids");
  }

  std::vector<int64_t> result;
  result.reserve(token_ids.size());
  for (const int32_t token_id : token_ids) {
    if (token_id < 0) {
      throw std::runtime_error("tokenizer.json produced a negative token id");
    }
    result.push_back(static_cast<int64_t>(token_id));
  }
  return result;
}

std::string DecodeWithTokenizerJson(const std::string& tokenizer_path,
                                    const std::vector<int64_t>& token_ids) {
  const std::string config = ReadTokenizerConfig(tokenizer_path);
  std::vector<int64_t> display_token_ids;
  display_token_ids.reserve(token_ids.size());
  for (const int64_t token_id : token_ids) {
    if (!AddedTokenIsSpecial(config, token_id)) {
      display_token_ids.push_back(token_id);
    }
  }
  if (display_token_ids.empty()) {
    return "";
  }

  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      LoadTokenizerJson(tokenizer_path);

  std::vector<int32_t> ids;
  ids.reserve(display_token_ids.size());
  for (const int64_t token_id : display_token_ids) {
    if (token_id < 0 || token_id > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error(
          "cannot decode token id outside int32 tokenizer range");
    }
    ids.push_back(static_cast<int32_t>(token_id));
  }
  return tokenizer->Decode(ids);
}

std::optional<int64_t> TokenIdForTokenizerString(
    const std::string& tokenizer_path, const std::string& token) {
  const std::optional<int64_t> config_id =
      AddedTokenIdFromConfig(ReadTokenizerConfig(tokenizer_path), token);
  if (config_id.has_value()) {
    return config_id;
  }
  if (token.starts_with("<")) {
    return std::nullopt;
  }

  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      LoadTokenizerJson(tokenizer_path);

  const int32_t id = tokenizer->TokenToId(token);
  if (id < 0) {
    return std::nullopt;
  }
  return static_cast<int64_t>(id);
}

TokenizerChatPrompt ApplyTokenizerChatTemplate(
    const std::string& tokenizer_path, const std::string& user_text) {
  const std::string chat_template =
      RequiredTokenizerConfigString(tokenizer_path, "chat_template");
  const std::string bos_token =
      TokenizerConfigString(tokenizer_path, "bos_token", "");

  if (Contains(chat_template, "<｜User｜>") &&
      Contains(chat_template, "<｜Assistant｜>")) {
    return TokenizerChatPrompt{
        bos_token + "<｜User｜>" + user_text + "<｜Assistant｜>",
        {"<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"}};
  }

  if (Contains(chat_template, "<start_of_turn>") &&
      Contains(chat_template, "<end_of_turn>")) {
    return TokenizerChatPrompt{bos_token + "<start_of_turn>user\n" + user_text +
                                   "<end_of_turn>\n<start_of_turn>model\n",
                               {"<end_of_turn>", "<start_of_turn>"}};
  }

  if (Contains(chat_template, "<|im_start|>") &&
      Contains(chat_template, "<|im_end|>")) {
    return TokenizerChatPrompt{"<|im_start|>user\n" + user_text +
                                   "<|im_end|>\n<|im_start|>assistant\n",
                               {"<|im_end|>", "<|im_start|>"}};
  }

  if (Contains(chat_template, "<|start_header_id|>") &&
      Contains(chat_template, "<|end_header_id|>")) {
    return TokenizerChatPrompt{
        bos_token + "<|start_header_id|>user<|end_header_id|>\n\n" + user_text +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        {"<|eot_id|>", "<|start_header_id|>"}};
  }

  if (Contains(chat_template, "<|user|>") &&
      Contains(chat_template, "<|assistant|>")) {
    return TokenizerChatPrompt{
        "<|user|>\n" + user_text + "<|end|>\n<|assistant|>\n",
        {"<|end|>", "<|user|>", "<|assistant|>"}};
  }

  throw std::runtime_error(
      "tokenizer_config.json chat_template is not supported by the native ONNX "
      "chat template adapter");
}

TokenizerChatPrompt ApplyTokenizerChatTemplate(
    const std::string& tokenizer_path,
    const std::vector<TokenizerChatMessage>& messages, bool add_assistant) {
  if (messages.empty()) {
    throw std::runtime_error("chat template requires at least one message");
  }

  const std::string chat_template =
      RequiredTokenizerConfigString(tokenizer_path, "chat_template");
  const std::string bos_token =
      TokenizerConfigString(tokenizer_path, "bos_token", "");

  const std::optional<std::string> literal_user_token =
      ExtractSingleQuotedLiteralContaining(chat_template, "User");
  const std::optional<std::string> literal_assistant_token =
      ExtractSingleQuotedLiteralContaining(chat_template, "Assistant");
  if (literal_user_token.has_value() && literal_assistant_token.has_value() &&
      Contains(chat_template, "message['content']")) {
    const std::string eos_token =
        TokenizerConfigString(tokenizer_path, "eos_token", "");
    std::string text = bos_token;
    for (const TokenizerChatMessage& message : messages) {
      if (message.role == "user") {
        text += *literal_user_token;
        text += message.content;
      } else if (message.role == "assistant") {
        text += *literal_assistant_token;
        text += message.content;
        text += eos_token;
      } else {
        throw std::runtime_error("unsupported chat message role: " +
                                 message.role);
      }
    }
    if (add_assistant) {
      text += *literal_assistant_token;
    }

    std::vector<std::string> stop_strings;
    if (!eos_token.empty()) {
      stop_strings.push_back(eos_token);
    }
    stop_strings.push_back(*literal_user_token);
    stop_strings.push_back(*literal_assistant_token);
    return TokenizerChatPrompt{std::move(text), std::move(stop_strings)};
  }

  if (Contains(chat_template, "<ï½œUserï½œ>") &&
      Contains(chat_template, "<ï½œAssistantï½œ>")) {
    std::string text = bos_token;
    for (const TokenizerChatMessage& message : messages) {
      AppendChatMessage(message, "<ï½œUserï½œ>", "<ï½œAssistantï½œ>", "",
                        &text);
    }
    if (add_assistant) {
      text += "<ï½œAssistantï½œ>";
    }
    return TokenizerChatPrompt{std::move(text),
                               {"<ï½œendâ–ofâ–sentenceï½œ>",
                                "<ï½œUserï½œ>", "<ï½œAssistantï½œ>"}};
  }

  if (Contains(chat_template, "<start_of_turn>") &&
      Contains(chat_template, "<end_of_turn>")) {
    std::string text = bos_token;
    for (const TokenizerChatMessage& message : messages) {
      AppendChatMessage(message, "<start_of_turn>user\n",
                        "<start_of_turn>model\n", "<end_of_turn>\n", &text);
    }
    if (add_assistant) {
      text += "<start_of_turn>model\n";
    }
    return TokenizerChatPrompt{std::move(text),
                               {"<end_of_turn>", "<start_of_turn>"}};
  }

  if (Contains(chat_template, "<|im_start|>") &&
      Contains(chat_template, "<|im_end|>")) {
    std::string text;
    for (const TokenizerChatMessage& message : messages) {
      AppendChatMessage(message, "<|im_start|>user\n",
                        "<|im_start|>assistant\n", "<|im_end|>\n", &text);
    }
    if (add_assistant) {
      text += "<|im_start|>assistant\n";
    }
    return TokenizerChatPrompt{std::move(text), {"<|im_end|>", "<|im_start|>"}};
  }

  if (Contains(chat_template, "<|start_header_id|>") &&
      Contains(chat_template, "<|end_header_id|>")) {
    std::string text = bos_token;
    for (const TokenizerChatMessage& message : messages) {
      AppendChatMessage(message, "<|start_header_id|>user<|end_header_id|>\n\n",
                        "<|start_header_id|>assistant<|end_header_id|>\n\n",
                        "<|eot_id|>", &text);
    }
    if (add_assistant) {
      text += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    return TokenizerChatPrompt{std::move(text),
                               {"<|eot_id|>", "<|start_header_id|>"}};
  }

  if (Contains(chat_template, "<|user|>") &&
      Contains(chat_template, "<|assistant|>")) {
    std::string text;
    for (const TokenizerChatMessage& message : messages) {
      AppendChatMessage(message, "<|user|>\n", "<|assistant|>\n", "<|end|>\n",
                        &text);
    }
    if (add_assistant) {
      text += "<|assistant|>\n";
    }
    return TokenizerChatPrompt{std::move(text),
                               {"<|end|>", "<|user|>", "<|assistant|>"}};
  }

  throw std::runtime_error(
      "tokenizer_config.json chat_template is not supported by the native ONNX "
      "chat template adapter");
}

TokenizerChatPrompt ApplyTokenizerChatTurnTemplate(
    const std::string& tokenizer_path, const std::string& user_text,
    bool first_turn) {
  const std::string chat_template =
      RequiredTokenizerConfigString(tokenizer_path, "chat_template");
  const std::string bos_token =
      TokenizerConfigString(tokenizer_path, "bos_token", "");

  if (Contains(chat_template, "<|start_header_id|>") &&
      Contains(chat_template, "<|end_header_id|>")) {
    return TokenizerChatPrompt{
        (first_turn ? bos_token : "<|eot_id|>") +
            std::string("<|start_header_id|>user<|end_header_id|>\n\n") +
            user_text +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        {"<|eot_id|>", "<|start_header_id|>"}};
  }

  if (Contains(chat_template, "<start_of_turn>") &&
      Contains(chat_template, "<end_of_turn>")) {
    return TokenizerChatPrompt{(first_turn ? bos_token : "<end_of_turn>\n") +
                                   std::string("<start_of_turn>user\n") +
                                   user_text +
                                   "<end_of_turn>\n<start_of_turn>model\n",
                               {"<end_of_turn>", "<start_of_turn>"}};
  }

  if (Contains(chat_template, "<|im_start|>") &&
      Contains(chat_template, "<|im_end|>")) {
    return TokenizerChatPrompt{
        (first_turn ? "" : "<|im_end|>\n") + std::string("<|im_start|>user\n") +
            user_text + "<|im_end|>\n<|im_start|>assistant\n",
        {"<|im_end|>", "<|im_start|>"}};
  }

  if (Contains(chat_template, "<|user|>") &&
      Contains(chat_template, "<|assistant|>")) {
    return TokenizerChatPrompt{(first_turn ? "" : "<|end|>\n") +
                                   std::string("<|user|>\n") + user_text +
                                   "<|end|>\n<|assistant|>\n",
                               {"<|end|>", "<|user|>", "<|assistant|>"}};
  }

  const std::optional<std::string> literal_user_token =
      ExtractSingleQuotedLiteralContaining(chat_template, "User");
  const std::optional<std::string> literal_assistant_token =
      ExtractSingleQuotedLiteralContaining(chat_template, "Assistant");
  if (literal_user_token.has_value() && literal_assistant_token.has_value() &&
      Contains(chat_template, "message['content']")) {
    const std::string eos_token =
        TokenizerConfigString(tokenizer_path, "eos_token", "");
    std::vector<std::string> stop_strings;
    if (!eos_token.empty()) {
      stop_strings.push_back(eos_token);
    }
    stop_strings.push_back(*literal_user_token);
    stop_strings.push_back(*literal_assistant_token);
    return TokenizerChatPrompt{(first_turn ? bos_token : eos_token) +
                                   *literal_user_token + user_text +
                                   *literal_assistant_token,
                               std::move(stop_strings)};
  }

  if (Contains(chat_template, "<ï½œUserï½œ>") &&
      Contains(chat_template, "<ï½œAssistantï½œ>")) {
    return TokenizerChatPrompt{
        (first_turn ? bos_token : "<ï½œendâ–ofâ–sentenceï½œ>") +
            std::string("<ï½œUserï½œ>") + user_text + "<ï½œAssistantï½œ>",
        {"<ï½œendâ–ofâ–sentenceï½œ>", "<ï½œUserï½œ>",
         "<ï½œAssistantï½œ>"}};
  }

  throw std::runtime_error(
      "tokenizer_config.json chat_template is not supported by the native ONNX "
      "chat template adapter");
}

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON
