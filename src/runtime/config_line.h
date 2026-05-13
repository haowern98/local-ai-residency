#ifndef MOSAICVRAM_SRC_RUNTIME_CONFIG_LINE_H_
#define MOSAICVRAM_SRC_RUNTIME_CONFIG_LINE_H_

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace mosaicvram {

struct ConfigLine {
  std::string kind;
  std::vector<std::string> positional;
  std::map<std::string, std::string> values;
  int line_number = 0;
};

std::string Trim(std::string_view text);
std::vector<std::string> TokenizeConfigLine(const std::string& line);
ConfigLine ParseConfigLine(const std::string& text, int line_number);

}  // namespace mosaicvram

#endif  // MOSAICVRAM_SRC_RUNTIME_CONFIG_LINE_H_
