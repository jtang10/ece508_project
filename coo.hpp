#pragma once

#include <functional>

#include "edge_list.hpp"
#include <vector>

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

template<typename Index>
using Vector = std::vector<Index>;

template <typename Index> class COO;

/*! \brief a read-only view of a COO suitable for passing to a GPU kernel by
value.

Any modifications to the underlying COO may invalidate this view.
*/
template <typename Index> class COOView {
  friend class COO<Index>;

private:
  uint64_t nnz_;      //!< number of non-zeros
  uint64_t num_rows_; //!< length of rowOffset - 1

public:
  typedef Index index_type;
  const Index *rowPtr_; //!< offset in col_ that each row starts at
  const Index *rowInd_; //!< non-zero row indices
  const Index *colInd_; //!< non-zero column indices

  HOST uint64_t nnz() const { return nnz_; }
  HOST uint64_t num_rows() const { return num_rows_; }

  const Index *row_ptr() const { return rowPtr_; }                    //!< row offset array
  const Index *col_ind() const { return colInd_; }                    //!< column index array
  const Index *row_ind() const { return rowInd_; }                    //<! row index array
  HOST DEVICE const Index *device_row_ptr() const { return rowPtr_; } //!< row offset array
  HOST DEVICE const Index *device_col_ind() const { return colInd_; } //!< column index array
  HOST DEVICE const Index *device_row_ind() const { return rowInd_; } //<! row index array
};

/*! \brief A COO matrix backed by CUDA Unified Memory, with a CSR rowPtr

Copying to a GPU kernel by value will cause the underling memory to be copied as
well. For read-only GPU access, use the view() method to get a lightweight
reference to the data.
*/
template <typename Index> class COO {
private:
  Index maxCol_;

public:
  typedef Index index_type;
  COO();                 //!< empty CSR
  Vector<Index> rowPtr_; //!< offset in col_ that each row starts at
  Vector<Index> colInd_; //!< non-zero column indices
  Vector<Index> rowInd_; //!< non-zero row indices
  HOST uint64_t nnz() const {
    assert(colInd_.size() == rowInd_.size());
    return colInd_.size();
  }                                      //!< number of non-zeros
  uint64_t num_nodes() const;            //!< number of unique row/col indices
  HOST uint64_t num_rows() const; //!< number of matrix rows


  /*! Build a COO from a sequence of edges

    Only include edges where f is true (default = all edges)

  */
  template <typename EdgeIter>
  static COO<Index> from_edges(EdgeIter begin, EdgeIter end,
                               std::function<bool(EdgeTy<Index>)> f = [](EdgeTy<Index> e) { return true; });

  COOView<Index> view() const; //!< create a COOView for this COO

  const Index *row_ptr() const { return rowPtr_.data(); } //!< row offset array
  const Index *col_ind() const { return colInd_.data(); } //!< column index array
  const Index *row_ind() const { return rowInd_.data(); } //<! row index array

  HOST DEVICE const Index *device_row_ptr() const { return rowPtr_.data(); } //!< row offset array
  HOST DEVICE const Index *device_col_ind() const { return colInd_.data(); } //!< column index array
  HOST DEVICE const Index *device_row_ind() const { return rowInd_.data(); } //<! row index array
};

#undef HOST
#undef DEVICE

#include "coo-impl.hpp"
