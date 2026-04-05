#include "cogstate/csv.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cogstate {

namespace {

std::vector<std::string> parse_csv_line(const std::string& line) {
  std::vector<std::string> out;
  std::string field;
  bool in_quotes = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"' && (i + 1 >= line.size() || line[i + 1] != '"')) {
      in_quotes = !in_quotes;
      continue;
    }
    if (ch == '"' && i + 1 < line.size() && line[i + 1] == '"') {
      field.push_back('"');
      ++i;
      continue;
    }
    if (ch == ',' && !in_quotes) {
      out.push_back(field);
      field.clear();
      continue;
    }
    field.push_back(ch);
  }
  out.push_back(field);
  return out;
}

std::string escape_csv_field(const std::string& value) {
  if (value.find_first_of(",\"\n") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (const char ch : value) {
    if (ch == '"') {
      escaped += "\"\"";
    } else {
      escaped.push_back(ch);
    }
  }
  escaped.push_back('"');
  return escaped;
}

}  // namespace

CsvTable read_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open CSV: " + path);
  }
  CsvTable table;
  std::string line;
  if (!std::getline(in, line)) {
    return table;
  }
  table.header = parse_csv_line(line);
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    table.rows.push_back(parse_csv_line(line));
  }
  return table;
}

void write_csv(const std::string& path, const CsvTable& table) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot write CSV: " + path);
  }
  for (std::size_t i = 0; i < table.header.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << escape_csv_field(table.header[i]);
  }
  out << "\n";
  for (const auto& row : table.rows) {
    for (std::size_t i = 0; i < row.size(); ++i) {
      if (i > 0) {
        out << ",";
      }
      out << escape_csv_field(row[i]);
    }
    out << "\n";
  }
}

}  // namespace cogstate

