#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cogstate/vector_db.hpp"

namespace {

std::vector<double> parse_vector(const std::string& csv) {
  std::vector<double> out;
  std::size_t start = 0;
  while (start <= csv.size()) {
    const std::size_t comma = csv.find(',', start);
    const std::string token =
        (comma == std::string::npos) ? csv.substr(start) : csv.substr(start, comma - start);
    if (!token.empty()) {
      out.push_back(std::stod(token));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace cogstate;
  if (argc < 3) {
    std::cerr << "Usage:\n"
              << "  cognitive_vector_db add <db_file> <id> <vec_csv> <metadata>\n"
              << "  cognitive_vector_db search <db_file> <vec_csv> <top_k>\n";
    return 1;
  }

  const std::string cmd = argv[1];
  VectorDatabase db;
  const std::string db_file = argv[2];
  try {
    db.load(db_file);
  } catch (...) {
  }

  if (cmd == "add") {
    if (argc < 6) {
      std::cerr << "Missing arguments for add\n";
      return 1;
    }
    const std::string id = argv[3];
    const Vector embedding = parse_vector(argv[4]);
    const std::string metadata = argv[5];
    db.upsert(id, embedding, metadata);
    db.save(db_file);
    std::cout << "Upserted state vector: " << id << "\n";
    return 0;
  }

  if (cmd == "search") {
    if (argc < 5) {
      std::cerr << "Missing arguments for search\n";
      return 1;
    }
    const Vector query = parse_vector(argv[3]);
    const std::size_t top_k = static_cast<std::size_t>(std::stoul(argv[4]));
    const auto results = db.semantic_state_search(query, top_k);
    for (const auto& r : results) {
      std::cout << r.id << "\t" << r.cosine_similarity << "\t" << r.metadata << "\n";
    }
    return 0;
  }

  std::cerr << "Unknown command: " << cmd << "\n";
  return 1;
}
