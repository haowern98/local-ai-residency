#include "runtime/config_line.h"

#include <cctype>
#include <sstream>
#include <stdexcept>

namespace mosaicvram {

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

namespace {

std::vector<std::string> TokenizeConfigLine(const std::string& line) {
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
  if (in_quotes) {
    throw std::runtime_error("unterminated quoted string");
  }
  if (!current.empty()) {
    tokens.push_back(current);
  }
  return tokens;
}

}  // namespace

ConfigLine ParseConfigLine(const std::string& text, int line_number) {
  const std::vector<std::string> tokens = TokenizeConfigLine(text);
  ConfigLine line;
  line.line_number = line_number;
  if (tokens.empty()) {
    return line;
  }
  line.kind = tokens.front();
  for (std::size_t i = 1; i < tokens.size(); ++i) {
    const std::size_t equals_pos = tokens[i].find('=');
    if (equals_pos == std::string::npos) {
      line.positional.push_back(tokens[i]);
      continue;
    }
    if (equals_pos == 0 || equals_pos + 1 >= tokens[i].size()) {
      std::ostringstream message;
      message << "line " << line_number << " has invalid key=value token";
      throw std::runtime_error(message.str());
    }
    line.values[tokens[i].substr(0, equals_pos)] =
        tokens[i].substr(equals_pos + 1);
  }
  return line;
}

}  // namespace mosaicvram
