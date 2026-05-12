#include "tokenizer/tokenizers_cpp_adapter.h"

#ifdef MOSAICVRAM_ENABLE_TOKENIZER_JSON

#include <tokenizers_cpp.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
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

}  // namespace

std::vector<int64_t> TokenizeWithTokenizerJson(
    const std::string& tokenizer_path, const std::string& prompt) {
  const std::filesystem::path json_path =
      ResolveTokenizerJsonPath(tokenizer_path);
  const std::string json_blob = ReadBinaryFile(json_path);
  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      tokenizers::Tokenizer::FromBlobJSON(json_blob);
  if (tokenizer == nullptr) {
    throw std::runtime_error("failed to load tokenizer.json: " +
                             json_path.string());
  }

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

}  // namespace mosaicvram

#endif  // MOSAICVRAM_ENABLE_TOKENIZER_JSON
