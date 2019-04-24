#pragma once

#include <cassert>
#include <string>
#include <cstring>

/*! check if base string ends with suffix string
/returns true if base ends with suffix, false otherwise
*/
bool endswith(const std::string &base,  //!< [in] the base string
              const std::string &suffix //!< [in] the suffix to check for
) {
  if (base.size() < suffix.size()) {
    return false;
  }
  return 0 == base.compare(base.size() - suffix.size(), suffix.size(), suffix);
}

/*! a class representing an edge list file
 */
class EdgeListFile {

private:
  enum class FileType { TSV, BEL };
  FILE *fp_;
  std::string path_;
  FileType type_;
  template <typename T> size_t read_bel(std::vector<std::pair<T,T>>& dest, const size_t n) {
    if (fp_ == nullptr) {
      return 0;
    }
    char *buf = new char[24 * n];
    const size_t numRead = fread(buf, 24, n, fp_);

    // end of file or error
    if (numRead != n) {
      // end of file
      if (feof(fp_)) {
        // do nothing
      }
      // some error
      else if (ferror(fp_)) {
        fclose(fp_);
        fp_ = nullptr;
        assert(0);
      } else {
        assert(0);
      }
    }
    for (size_t i = 0; i < numRead; ++i) {
      uint64_t src, dst;
      std::memcpy(&src, &buf[i * 24 + 8], 8);
      std::memcpy(&dst, &buf[i * 24 + 0], 8);
      dest[i].first = src;
      dest[i].second = dst;
    }

    // no characters extracted or parsing error
    delete[] buf;
    return numRead;
  }

  template <typename T> size_t read_tsv(std::vector<std::pair<T,T>>& dest, const size_t n) {

    assert(fp_ != nullptr);

    size_t i = 0;
    for (; i < n; ++i) {
      long long unsigned dst, src, weight;
      const size_t numFilled = fscanf(fp_, "%llu %llu %llu", &dst, &src, &weight);
      if (numFilled != 3) {
        if (feof(fp_)) {
          return i;
        } else if (ferror(fp_)) {
          return i;
        } else {
          exit(-1);
        }
      }
      dest[i].first = static_cast<T>(src);
      dest[i].second = static_cast<T>(dst);
    }
    return i;
  }

public:
  /*! \brief Construct an EdgeListFile
    Supports GraphChallenge TSV or BEL files
  */
  EdgeListFile(const std::string &path //!< [in] the path of the file
               )
      : path_(path) {
    if (endswith(path, ".bel")) {
      type_ = FileType::BEL;
    } else if (endswith(path, ".tsv")) {
      type_ = FileType::TSV;
    } else {
      exit(-1);
    }

    fp_ = fopen(path_.c_str(), "r");
    if (nullptr == fp_) {
    }
  }

  ~EdgeListFile() {
    if (fp_) {
      fclose(fp_);
      fp_ = nullptr;
    }
  }

  /*! \brief attempt to read n edges from the file
    \tparam T the node ID type
    \returns the number of edges read
  */
  template <typename T>
  size_t
  get_edges(std::vector<std::pair<T,T>> &edges, //!< [out] the read edges. Resized to the number of successfully read edges
            const size_t n                 //!< [in] the number of edges to try to read
  ) {
    edges.resize(n);

    size_t numRead;
    switch (type_) {
    case FileType::BEL: {
      numRead = read_bel(edges, n);
      break;
    }
    case FileType::TSV: {
      numRead = read_tsv(edges, n);
      break;
    }
    default: {
      exit(-1);
    }
    }
    edges.resize(numRead);
    return numRead;
  }
};
