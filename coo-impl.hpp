#pragma once

#include <set>
#include <cassert>
#include "coo.hpp"

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

template <typename Index> COO<Index>::COO() {}

template <typename Index> HOST uint64_t COO<Index>::num_rows() const {
  if (rowPtr_.size() == 0) {
    return 0;
  } else {
    return rowPtr_.size() - 1;
  }
}

template <typename Index> uint64_t COO<Index>::num_nodes() const {
  std::set<Index> nodes;
  // add all dsts
  for (Index ci = 0; ci < colInd_.size(); ++ci) {
    nodes.insert(colInd_[ci]);
  }
  // add non-zero sources
  for (Index i = 0; i < rowPtr_.size() - 1; ++i) {
    Index row_start = rowPtr_[i];
    Index row_end = rowPtr_[i + 1];
    if (row_start != row_end) {
      nodes.insert(i);
    }
  }
  return nodes.size();
}

/*! Build a COO from a sequence of edges

*/
template <typename Index>
template <typename EdgeIter>
COO<Index> COO<Index>::from_edges(EdgeIter begin, EdgeIter end, std::function<bool(EdgeTy<Index>)> f) {
  COO<Index> coo;

  if (begin == end) {
    return coo;
  }

  for (auto ei = begin; ei != end; ++ei) {
    EdgeTy<Index> edge = *ei;
    const Index src = edge.first;
    const Index dst = edge.second;

    // edge has a new src and should be in a new row
    // even if the edge is filtered out, we need to add empty rows
    while (coo.rowPtr_.size() != size_t(src + 1)) {
      // expecting inputs to be sorted by src, so it should be at least
      // as big as the current largest row we have recored
      assert(src >= coo.rowPtr_.size() && "are edges not ordered by source?");
      coo.rowPtr_.push_back(coo.colInd_.size());
    }

    if (f(edge)) {
      coo.rowInd_.push_back(src);
      coo.colInd_.push_back(dst);
    } else {
      continue;
    }
  }

  // add the final length of the non-zeros to the offset array
  coo.rowPtr_.push_back(coo.colInd_.size());

  assert(coo.rowInd_.size() == coo.colInd_.size());
  return coo;
}

template<typename Index>
 COOView<Index> COO<Index>::view() const {
     COOView<Index> view;
     view.nnz_ = nnz();
     view.num_rows_ = num_rows();
     view.rowPtr_ = rowPtr_.data();
     view.colInd_ = colInd_.data();
     view.rowInd_ = rowInd_.data();
     return view;
 }
 


#undef HOST
#undef DEVICE
