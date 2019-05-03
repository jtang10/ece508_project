#include <vector>
#include <list>
#include <cassert>
#include <algorithm>
#include "util.cpp"
#include "edge_list_file.hpp"
#include "coo-impl.hpp"
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

__global__ static void count_triangle(uint64_t *triangleCount, //!< per-edge triangle counts
                                      const uint64_t *const edgeSrc,         //!< node ids for edge srcs
                                      const uint64_t *const edgeDst,         //!< node ids for edge dsts
                                      const uint64_t *const rowPtr,          //!< source node offsets in edgeDst
                                      const uint64_t numEdges                  //!< how many edges to count triangles for
) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for(uint64_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    // Determine the source and destination node for the edge
    uint64_t u = edgeSrc[idx];
    uint64_t v = edgeDst[idx];

    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    uint64_t u_ptr = rowPtr[u];
    uint64_t v_ptr = rowPtr[v];

    uint64_t u_end = rowPtr[u + 1];
    uint64_t v_end = rowPtr[v + 1];

    uint64_t w1 = edgeDst[u_ptr];
    uint64_t w2 = edgeDst[v_ptr];

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
  }
}

__global__ static void write_triangle(uint64_t *triangleOffsets, //!< per-edge triangle offsets
                                      pair<uint64_t,uint64_t> *triangleBuffers, //!< per-edge triangle buffers
                                      const uint64_t *const edgeSrc,         //!< node ids for edge srcs
                                      const uint64_t *const edgeDst,         //!< node ids for edge dsts
                                      const uint64_t *const rowPtr,          //!< source node offsets in edgeDst
                                      const uint64_t numEdges                  //!< how many edges to count triangles for
) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(uint64_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    uint64_t local_offset = triangleOffsets[i];
    // Determine the source and destination node for the edge
    uint64_t u = edgeSrc[idx];
    uint64_t v = edgeDst[idx];

    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    uint64_t u_ptr = rowPtr[u];
    uint64_t v_ptr = rowPtr[v];

    uint64_t u_end = rowPtr[u + 1];
    uint64_t v_end = rowPtr[v + 1];

    uint64_t w1 = edgeDst[u_ptr];
    uint64_t w2 = edgeDst[v_ptr];

    // Determine how many elements of those two arrays are common
    while ((u_ptr < u_end) && (v_ptr < v_end)) {
      if (w1 < w2) {
        w1 = edgeDst[++u_ptr];
      } else if (w1 > w2) {
        w2 = edgeDst[++v_ptr];
      } else {
        w1 = edgeDst[++u_ptr];
        w2 = edgeDst[++v_ptr];
        triangleBuffers[local_offset++]=make_pair<uint64_t,uint64_t>(u_ptr,v_ptr);
      }
    }
  }
}

__global__ static void truss_decompose(uint64_t* triangleCounts, //!< per-edge triangle counts
                                       uint64_t* triangleOffsets, //!< per-edge triangle offsets
                                       uint64_t* triangleRemove, //!< per-edge triangle removes
                                       pair<uint64_t,uint64_t>* triangleBuffers, //!< per-edge triangle buffer
                                       int* edge_exists,
                                       int* new_deletes
) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local_edge_exists=0;
  int local_new_deletes=0;
  for (uint64_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    uint64_t my_triangle_count=triangleCounts[i];
    if (my_triangle_count<(k-2)*3&&my_triangle_count>0) { //remove edge
      local_new_deletes=1;
      atomicAdd(triangleRemove+i,my_triangle_count);
      for (uint64_t iter = triangleOffsets[i];iter!=triangleOffsets[i+1];++iter) {
        atomicAdd(triangleRemove+triangleBuffers[iter].first,3);
        atomicAdd(triangleRemove+triangleBuffers[iter].second,3);
      }
    }
    else if (my_triangle_count>0) {
      local_edge_exists=1;
    }
  }
  atomicCAS(edge_exists,0,local_edge_exists);
  atomicCAS(new_deletes,0,local_new_deletes)
}

__global__ static void update_triangles(uint64_t* triangleCounts, //!< per-edge triangle counts
                                        uint64_t* triangleOffsets, //!< per-edge triangle offsets
                                        uint64_t* triangleRemove, //!< per-edge triangle removes
                                        pair<uint64_t,uint64_t>* triangleBuffers, //!< per-edge triangle buffer
) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint64_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    if (triangleCounts[i]>triangleRemove[i]) { //must remove triangles associated with this edge
      for (uint64_t iter = triangleOffsets[i];iter!=triangleOffsets[i+1];++iter) {
        uint64_t e1=atomicAdd(&triangleBuffers[iter].first,0);
        uint64_t e2=atomicAdd(&triangleBuffers[iter].second,0);
        if (e1!=-1) {
          for (uint64_t j=triangleOffsets[e1];j!=triangleOffsets[e1+1];++j) {
            uint64_t ea=atomicAdd(&triangleBuffers[j].first,0);
            uint64_t eb=atomicAdd(&triangleBuffers[j].second,0);
            if (ea==i||eb==i) {
              atomicExch(&triangleBuffers[j].first,-1);
              atomicExch(&triangleBuffers[j].second,-1);
            }
          }
        }
        if (e2!=-1) {
          for (uint64_t j=triangleOffsets[e2];j!=triangleOffsets[e2+1];++j) {
            uint64_t ea=atomicAdd(&triangleBuffers[j].first,0);
            uint64_t eb=atomicAdd(&triangleBuffers[j].second,0);
            if (ea==i||eb==i) {
              atomicExch(&triangleBuffers[j].first,-1);
              atomicExch(&triangleBuffers[j].second,-1);
            }
          }
        }
      }
    }
    else {
      triangleCounts[i]-=triangleRemove[i];
    }
  }
}

int main(int argc, char * argv[]) {
  std::string test_filename;	
  if (argv[1] == NULL) {
      test_filename = "./data/test3.bel";
  } else {
    test_filename = argv[1];
  }
  EdgeListFile test_file(test_filename);

  // get the total number of edges in the file.
  std::vector<EdgeTy<uint64_t>> edges;
  uint64_t size = getNumEdges(test_filename);
  std::cout << "Numbers of edges in the file : " << size << std::endl;

  // read the bel file into the EdgeListFile
  uint64_t numEdge = test_file.get_edges(edges, size);
  std::cout << "Confirmed read edges: " << numEdge << std::endl;
  
  COO<uint64_t> coo_test = COO<uint64_t>::from_edges<std::vector<EdgeTy<uint64_t>>::iterator>(edges.begin(), edges.end());
  COOView<uint64_t> test_view = coo_test.view();
  
  uint64_t numEdges = test_view.nnz();
  uint64_t numRows = test_view.num_rows();
  // vector<uint64_t> triangleCount(numEdges);  // keep track of the number of triangles for each edge
  // vector<vector<pair<uint64_t, uint64_t>>> triangleList(numEdges); // keep track of the triangle edges for each edge
  std::cout << "numEdges from nnz: " << numEdges << std::endl;

	uint64_t* edgeSrc_device = nullptr;
  uint64_t* edgeDst_device = nullptr;
	uint64_t* rowPtr_device = nullptr;
	uint64_t* triangleCount = nullptr;
  uint64_t* triangleOffsets = nullptr;
  uint64_t* triangleBuffers = nullptr;
  uint64_t* triangleRemove = nullptr;
  int* edge_exists = nullptr;
  int* new_deletes = nullptr;
  int k=2;

	//allocate necessary memory
	cudaMalloc(&edgeSrc_device, numEdges*sizeof(uint64_t));
	cudaMalloc(&edgeDst_device, numEdges*sizeof(uint64_t));
	cudaMalloc(&rowPtr_device, (numRows+1)*sizeof(uint64_t));
	cudaMallocManaged(&triangleCount, numEdges*sizeof(uint64_t));
  cudaMallocManaged(&triangleOffsets, (numEdges+1)*sizeof(uint64_t));
  cudaMallocManaged(&triangleRemove, numEdges*sizeof(uint64_t));
  cudaMallocManaged(&edge_exists, sizeof(int));
  cudaMallocManaged(&new_deletes, sizeof(int));
  *edge_exists=1;
  *new_deletes=0;
	//copy over data
	cudaMemcpy(edgeSrc_device, test_view.row_ind(), numEdges*sizeof(uint64_t),cudaMemcpyHostToDevice);
	cudaMemcpy(edgeDst_device, test_view.col_ind(), numEdges*sizeof(uint64_t),cudaMemcpyHostToDevice);
	cudaMemcpy(rowPtr_device, test_view.row_ptr(), (numRows+1)*sizeof(uint64_t),cudaMemcpyHostToDevice);
  //call triangle_count
  dim3 dimBlock(512);
  dim3 dimGrid (ceil(numEdges * 1.0 / dimBlock.x));
	count_triangle<<<dimBlock, dimGrid>>>(triangleCount, edgeSrc_device, edgeDst_device, rowPtr_device, numEdges);
	cudaDeviceSynchronize();
  thrust::device_ptr<uint64_t> triangleCount_ptr(triangleCount);
  triangleOffsets[0]=0;
  thrust::device_ptr<uint64_t> triangleOffsets_ptr(triangleOffsets);
  thrust::inclusive_scan(triangleCount_ptr,triangleCount_ptr+numEdges,triangleOffsets_ptr+1);
  cudaDeviceSynchronize();
  cudaMallocManaged(&triangleBuffers,triangleOffsets[numEdges]*sizeof(pair<uint64_t,uint64_t>));
  write_triangle<<<dimBlock, dimGrid>>>(triangleOffsets, triangleBuffers, edgeSrc_device, edgeDst_device, rowPtr_device, numEdges);
  cudaDeviceSynchronize();
  while (*edge_exists) {
    if (*new_deletes==0) {
      ++k;
    }
    *edge_exists=0;
    *new_deletes=0;
    thrust::device_ptr<uint64_t> triangleRemove_ptr(triangleRemove);
    thrust::fill(triangleRemove_ptr,triangleRemove_ptr+numEdges,0);
    cudaDeviceSynchronize();
    truss_decompose<<<dimBlock, dimGrid>>>(triangleCount, triangleOffsets, triangleRemove, triangleBuffers, edge_exists, new_deletes);
    cudaDeviceSynchronize();
    if (*new_deletes==1) {
      update_triangles<<<dimBlock, dimGrid>>>(triangleCount, triangleOffsets, triangleRemove, triangleBuffers);
    }
  }
	std::cout << "Triangle Count" << std::endl;
	uint64_t totalCount = 0;
	for (uint64_t i = 0; i < numEdges; ++i) {
		std::cout << triangleCount[i] << ' ';
		totalCount += triangleCount[i];
	}
  std::cout << "totalCount: " << totalCount << std::endl;
  
  cudaFree(edgeSrc_device);
  cudaFree(edgeDst_device);
  cudaFree(rowPtr_device);
  cudaFree(triangleCount);
  cudaFree(triangleOffsets);
  return 0;
}
