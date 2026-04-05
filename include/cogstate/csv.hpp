#pragma once

#include <string>
#include <vector>

namespace cogstate {

class CsvTable {
 public:
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
};

CsvTable read_csv(const std::string& path);
void write_csv(const std::string& path, const CsvTable& table);

}  // namespace cogstate

