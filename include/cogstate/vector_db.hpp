#pragma once

#include <string>
#include <vector>

#include "cogstate/types.hpp"

namespace cogstate {

class VectorDatabase {
 public:
  void upsert(std::string id, Vector embedding, std::string metadata);
  std::vector<SearchResult> semantic_state_search(const Vector& query, std::size_t top_k) const;
  void save(const std::string& path) const;
  void load(const std::string& path);

 private:
  struct Entry {
    std::string id;
    Vector embedding;
    std::string metadata;
  };
  std::vector<Entry> entries_;
};

}  // namespace cogstate

