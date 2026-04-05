#include "cogstate/vector_db.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace cogstate {

namespace {

double norm(const Vector& x) {
  return std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0));
}

double cosine(const Vector& a, const Vector& b) {
  if (a.size() != b.size() || a.empty()) {
    return -1.0;
  }
  const double na = norm(a);
  const double nb = norm(b);
  if (na <= 1e-12 || nb <= 1e-12) {
    return -1.0;
  }
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0) / (na * nb);
}

}  // namespace

void VectorDatabase::upsert(std::string id, Vector embedding, std::string metadata) {
  auto it = std::find_if(entries_.begin(), entries_.end(),
                         [&id](const Entry& e) { return e.id == id; });
  if (it == entries_.end()) {
    entries_.push_back({std::move(id), std::move(embedding), std::move(metadata)});
    return;
  }
  it->embedding = std::move(embedding);
  it->metadata = std::move(metadata);
}

std::vector<SearchResult> VectorDatabase::semantic_state_search(const Vector& query,
                                                                std::size_t top_k) const {
  std::vector<SearchResult> results;
  results.reserve(entries_.size());
  for (const auto& e : entries_) {
    results.push_back({e.id, cosine(query, e.embedding), e.metadata});
  }
  std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
    return a.cosine_similarity > b.cosine_similarity;
  });
  if (results.size() > top_k) {
    results.resize(top_k);
  }
  return results;
}

void VectorDatabase::save(const std::string& path) const {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot save vector DB: " + path);
  }
  for (const auto& e : entries_) {
    out << e.id << "\t" << e.metadata << "\t";
    for (std::size_t i = 0; i < e.embedding.size(); ++i) {
      if (i > 0) {
        out << ",";
      }
      out << e.embedding[i];
    }
    out << "\n";
  }
}

void VectorDatabase::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot load vector DB: " + path);
  }
  entries_.clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    const std::size_t t1 = line.find('\t');
    const std::size_t t2 = (t1 == std::string::npos) ? std::string::npos : line.find('\t', t1 + 1);
    if (t1 == std::string::npos || t2 == std::string::npos) {
      continue;
    }
    Entry e;
    e.id = line.substr(0, t1);
    e.metadata = line.substr(t1 + 1, t2 - t1 - 1);
    std::string vals = line.substr(t2 + 1);
    std::size_t start = 0;
    while (start <= vals.size()) {
      const std::size_t comma = vals.find(',', start);
      const std::string token = (comma == std::string::npos) ? vals.substr(start)
                                                              : vals.substr(start, comma - start);
      if (!token.empty()) {
        e.embedding.push_back(std::stod(token));
      }
      if (comma == std::string::npos) {
        break;
      }
      start = comma + 1;
    }
    entries_.push_back(std::move(e));
  }
}

}  // namespace cogstate

