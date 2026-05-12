#include "tokenizer/hf_tokenizer_adapter.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace mosaicvram {
namespace {

int CurrentProcessId() {
#ifdef _WIN32
  return _getpid();
#else
  return getpid();
#endif
}

std::filesystem::path MakeTempDirectory() {
  const std::filesystem::path base = std::filesystem::temp_directory_path();
  for (int attempt = 0; attempt < 100; ++attempt) {
    const std::filesystem::path path =
        base / ("mosaicvram_tokenizer_" + std::to_string(CurrentProcessId()) +
                "_" + std::to_string(attempt));
    std::error_code error;
    if (std::filesystem::create_directory(path, error)) {
      return path;
    }
  }
  throw std::runtime_error("failed to create tokenizer temp directory");
}

void WriteTextFile(const std::filesystem::path& path,
                   const std::string& text) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to write tokenizer temp file: " +
                             path.string());
  }
  output << text;
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to read tokenizer output file: " +
                             path.string());
  }
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

std::vector<int64_t> ParseTokenList(std::string_view text) {
  std::vector<int64_t> tokens;
  std::size_t begin = 0;
  while (begin < text.size()) {
    while (begin < text.size() &&
           (text[begin] == ',' || text[begin] == ' ' ||
            text[begin] == '\n' || text[begin] == '\r' ||
            text[begin] == '\t')) {
      ++begin;
    }
    if (begin >= text.size()) {
      break;
    }
    std::size_t end = begin;
    while (end < text.size() && text[end] != ',' && text[end] != ' ' &&
           text[end] != '\n' && text[end] != '\r' && text[end] != '\t') {
      ++end;
    }
    const std::string token_text(text.substr(begin, end - begin));
    std::size_t parsed_chars = 0;
    const int64_t token = std::stoll(token_text, &parsed_chars);
    if (parsed_chars != token_text.size() || token < 0) {
      throw std::runtime_error("tokenizer produced invalid token id");
    }
    tokens.push_back(token);
    begin = end;
  }
  if (tokens.empty()) {
    throw std::runtime_error("tokenizer produced no token ids");
  }
  return tokens;
}

std::string TokenizerScript() {
  return R"PY(import argparse
from pathlib import Path
from transformers import AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()
parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

prompt = Path(args.input).read_text(encoding="utf-8")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
token_ids = tokenizer.encode(prompt, add_special_tokens=False)
Path(args.output).write_text(",".join(str(token_id) for token_id in token_ids),
                             encoding="ascii")
)PY";
}

void RunTokenizerProcess(const HfTokenizerOptions& options,
                         const std::filesystem::path& script_path,
                         const std::filesystem::path& prompt_path,
                         const std::filesystem::path& output_path) {
#ifdef _WIN32
  std::string script = script_path.string();
  std::string prompt = prompt_path.string();
  std::string output = output_path.string();
  const char* argv[] = {options.python_executable.c_str(),
                        script.c_str(),
                        "--tokenizer",
                        options.tokenizer_path.c_str(),
                        "--input",
                        prompt.c_str(),
                        "--output",
                        output.c_str(),
                        nullptr};
  const intptr_t exit_code =
      _spawnvp(_P_WAIT, options.python_executable.c_str(), argv);
  if (exit_code != 0) {
    throw std::runtime_error(
        "failed to tokenize ONNX prompt; ensure Python, transformers, and "
        "the model tokenizer files are available");
  }
#else
  const std::string command =
      "'" + options.python_executable + "' '" + script_path.string() +
      "' --tokenizer '" + options.tokenizer_path + "' --input '" +
      prompt_path.string() + "' --output '" + output_path.string() + "'";
  const int exit_code = std::system(command.c_str());
  if (exit_code != 0) {
    throw std::runtime_error(
        "failed to tokenize ONNX prompt; ensure Python, transformers, and "
        "the model tokenizer files are available");
  }
#endif
}

}  // namespace

std::vector<int64_t> TokenizeWithHfTokenizer(
    const HfTokenizerOptions& options, const std::string& prompt) {
  if (options.tokenizer_path.empty()) {
    throw std::runtime_error("tokenizer path is required");
  }
  const std::filesystem::path temp_dir = MakeTempDirectory();
  const std::filesystem::path script_path = temp_dir / "tokenize.py";
  const std::filesystem::path prompt_path = temp_dir / "prompt.txt";
  const std::filesystem::path output_path = temp_dir / "tokens.txt";

  try {
    WriteTextFile(script_path, TokenizerScript());
    WriteTextFile(prompt_path, prompt);

    RunTokenizerProcess(options, script_path, prompt_path, output_path);

    const std::vector<int64_t> tokens =
        ParseTokenList(ReadTextFile(output_path));
    std::error_code cleanup_error;
    std::filesystem::remove_all(temp_dir, cleanup_error);
    return tokens;
  } catch (...) {
    std::error_code cleanup_error;
    std::filesystem::remove_all(temp_dir, cleanup_error);
    throw;
  }
}

}  // namespace mosaicvram
