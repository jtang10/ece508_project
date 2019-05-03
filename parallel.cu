#include <vector>
#include <list>
#include <cassert>
#include <algorithm>
#include "util.cpp"
#include "edge_list_file.hpp"
#include "coo-impl.hpp"
#include <chrono>
#include <thrust/scan.h>

__global__ static void count_triangle(uint64_t *triangleCount,        //!< per-edge triangle counts
                                      const uint32_t *const edgeSrc,  //!< node ids for edge srcs
                                      const uint32_t *const edgeDst,  //!< node ids for edge dsts
                                      const uint32_t *const rowPtr,   //!< source node offsets in edgeDst
                                      const uint32_t numEdges         //!< how many edges to count triangles for
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for(uint32_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    // Determine the source and destination node for the edge
    uint32_t u = edgeSrc[idx];
    uint32_t v = edgeDst[idx];

    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    uint32_t u_ptr = rowPtr[u];
    uint32_t v_ptr = rowPtr[v];

    uint32_t u_end = rowPtr[u + 1];
    uint32_t v_end = rowPtr[v + 1];

    uint32_t w1 = edgeDst[u_ptr];
    uint32_t w2 = edgeDst[v_ptr];

    // Determine how many elements of those two arrays are common
    while ((u_ptr < u_end) && (v_ptr < v_end)) {
      if (w1 < w2) {
        w1 = edgeDst[++u_ptr];
      } else if (w1 > w2) {
        w2 = edgeDst[++v_ptr];
      } else {
        w1 = edgeDst[++u_ptr];
        w2 = edgeDst[++v_ptr];
        triangleCount[idx]++;
      }
    }

    // if (triangleCount[idx]) {
    //   printf("Thread %d output: %d", threadIdx.x, triangleCount[idx]);
    // }
  }
}

__global__ static void enum_triangle()

int main(int argc, char * argv[]) {
  std::string test_filename;	
  if (argv[1] == NULL) {
      test_filename = "./data/test3.bel";
  } else {
    test_filename = argv[1];
  }
  EdgeListFile test_file(test_filename);

  // get the total number of edges in the file.
  std::vector<EdgeTy<uint32_t>> edges;
  uint32_t size = getNumEdges(test_filename);
  // std::cout << "Numbers of edges in the file : " << size << std::endl;

  // read the bel file into the EdgeListFile
  uint32_t numEdge = test_file.get_edges(edges, size);
  // std::cout << "Confirmed read edges: " << numEdge << std::endl;
  
  COO<uint32_t> coo_test = COO<uint32_t>::from_edges<std::vector<EdgeTy<uint32_t>>::iterator>(edges.begin(), edges.end());
  COOView<uint32_t> test_view = coo_test.view();
  
  uint32_t numEdges = test_view.nnz();
  uint32_t numRows = test_view.num_rows();
  // vector<uint32_t> triangleCount(numEdges);  // keep track of the number of triangles for each edge
  // vector<vector<pair<uint32_t, uint32_t>>> triangleList(numEdges); // keep track of the triangle edges for each edge
  assert(size == numEdge && "number of edges from bel file size does not equal confirmed read edges.");
  assert(numEdge == numEdges && "number of edges in COO does not equal confirmed read edges");
  std::cout << "numEdges from nnz: " << numEdges << std::endl;


	uint32_t* edgeSrc_device = nullptr;
  uint32_t* edgeDst_device = nullptr;
	uint32_t* rowPtr_device = nullptr;
	uint64_t* triangleCount = nullptr;
	uint64_t* triangleCountScan = nullptr;

	//allocate necessary memory
	cudaMalloc(&edgeSrc_device, numEdges*sizeof(uint32_t));
	cudaMalloc(&edgeDst_device, numEdges*sizeof(uint32_t));
	cudaMalloc(&rowPtr_device, (numRows+1)*sizeof(uint32_t));
	cudaMallocManaged(&triangleCount, numEdges*sizeof(uint64_t));
	cudaMallocManaged(&triangleCountScan, numEdges*sizeof(uint64_t));
	//copy over data
	cudaMemcpy(edgeSrc_device, test_view.row_ind(), numEdges*sizeof(uint32_t),cudaMemcpyHostToDevice);
	cudaMemcpy(edgeDst_device, test_view.col_ind(), numEdges*sizeof(uint32_t),cudaMemcpyHostToDevice);
	cudaMemcpy(rowPtr_device, test_view.row_ptr(), (numRows+1)*sizeof(uint32_t),cudaMemcpyHostToDevice);
  //call triangle_count
  // chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  dim3 dimBlock(512);
  dim3 dimGrid (ceil(numEdges * 1.0 / dimBlock.x));
	count_triangle<<<dimBlock, dimGrid>>>(triangleCount, edgeSrc_device, edgeDst_device, rowPtr_device, numEdges);
	cudaDeviceSynchronize();
  // chrono::steady_clock::time_point end= chrono::steady_clock::now();
  // std::cout << "Triangle count time = " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << " us" << std::endl;
	std::cout << "Triangle Count" << std::endl;
	uint64_t totalCount = 0;
	for (uint32_t i = 0; i < numEdges; ++i) {
		std::cout << triangleCount[i] << ' ';
		totalCount += triangleCount[i];
	}
  std::cout << "totalCount: " << totalCount << std::endl;

  thrust::inclusive_scan(triangleCount, triangleCount+numEdges, triangleCountScan);
  std::cout << "Triangle Count Scan" << std::endl;
	for (uint32_t i = 0; i < numEdges; ++i) {
		std::cout << triangleCountScan[i] << ' ';
	}
  
  cudaFree(edgeSrc_device);
  cudaFree(edgeDst_device);
  cudaFree(rowPtr_device);
  cudaFree(triangleCount);

  return 0;
}
