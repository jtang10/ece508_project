#include <vector>
#include <list>
#include <cassert>
#include <numeric>
#include <algorithm>
#include "util.cpp"
#include "edge_list_file.hpp"
#include "coo-impl.hpp"
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <utility>
#include <iostream> 

__global__ static void count_triangle(int32_t *triangleCount, //!< per-edge triangle counts
                                      const int32_t *const edgeSrc,         //!< node ids for edge srcs
                                      const int32_t *const edgeDst,         //!< node ids for edge dsts
                                      const int32_t *const rowPtr,          //!< source node offsets in edgeDst
                                      const int32_t numEdges                  //!< how many edges to count triangles for
) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for(int32_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    // Determine the source and destination node for the edge
    int32_t u = edgeSrc[idx];
    int32_t v = edgeDst[idx];

    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    int32_t u_ptr = rowPtr[u];
    int32_t v_ptr = rowPtr[v];

    int32_t u_end = rowPtr[u + 1];
    int32_t v_end = rowPtr[v + 1];

    int32_t w1 = edgeDst[u_ptr];
    int32_t w2 = edgeDst[v_ptr];

    // Determine how many elements of those two arrays are common
    while ((u_ptr < u_end) && (v_ptr < v_end)) {
      if (w1 < w2) {
        w1 = edgeDst[++u_ptr];
      } else if (w1 > w2) {
        w2 = edgeDst[++v_ptr];
      } else {
        atomicAdd(triangleCount+idx, 1);
        atomicAdd(triangleCount+u_ptr, 1);
        atomicAdd(triangleCount+v_ptr, 1);
        w1 = edgeDst[++u_ptr];
        w2 = edgeDst[++v_ptr];
      }
    }
  }
}

__global__ static void write_triangle(int32_t *triangleOffsets, //!< per-edge triangle offsets
                                      int32_t *triangleOffCounts,
                                      int32_t *triangleBuffers1, //!< per-edge triangle buffers
                                      int32_t *triangleBuffers2, 
                                      const int32_t *const edgeSrc,         //!< node ids for edge srcs
                                      const int32_t *const edgeDst,         //!< node ids for edge dsts
                                      const int32_t *const rowPtr,          //!< source node offsets in edgeDst
                                      const int32_t numEdges                  //!< how many edges to count triangles for
) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int32_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    // Determine the source and destination node for the edge
    int32_t u = edgeSrc[idx];
    int32_t v = edgeDst[idx];

    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    int32_t u_ptr = rowPtr[u];
    int32_t v_ptr = rowPtr[v];

    int32_t u_end = rowPtr[u + 1];
    int32_t v_end = rowPtr[v + 1];

    int32_t w1 = edgeDst[u_ptr];
    int32_t w2 = edgeDst[v_ptr];

    // Determine how many elements of those two arrays are common
    while ((u_ptr < u_end) && (v_ptr < v_end)) {
      if (w1 < w2) {
        w1 = edgeDst[++u_ptr];
      } else if (w1 > w2) {
        w2 = edgeDst[++v_ptr];
      } else {
        int32_t local_offset=atomicAdd(triangleOffCounts+i,1);
        triangleBuffers1[local_offset]=u_ptr;
        triangleBuffers2[local_offset]=v_ptr;
        int32_t u_offset=atomicAdd(triangleOffCounts+u_ptr,1);
        triangleBuffers1[u_offset]=v_ptr;
        triangleBuffers2[u_offset]=i;
        int32_t v_offset=atomicAdd(triangleOffCounts+v_ptr,1);
        triangleBuffers1[v_offset]=u_ptr;
        triangleBuffers2[v_offset]=i;
        w1 = edgeDst[++u_ptr];
        w2 = edgeDst[++v_ptr];
      }
    }
  }
}

__global__ static void truss_decompose(int32_t* triangleCounts, //!< per-edge triangle counts
                                       int32_t* triangleOffsets, //!< per-edge triangle offsets
                                       int32_t* removeFlag,
                                       int32_t* triangleBuffers1, //!< per-edge triangle buffer
                                       int32_t* triangleBuffers2, //!< per-edge triangle buffer
                                       int* edge_exists,
                                       int* new_deletes,
                                       int32_t numEdges,
                                       int k
) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local_edge_exists=0;
  int local_new_deletes=0;
  for (int32_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    int32_t my_triangle_count=triangleCounts[i];
    if (my_triangle_count<(k-2)*3&&my_triangle_count>0) { //remove edge
      local_new_deletes=1;
      removeFlag[i] = 1;
    }
    else if (my_triangle_count>(0)) {
      local_edge_exists=1;
    }
  }
  atomicCAS(edge_exists,0,local_edge_exists);
  atomicCAS(new_deletes,0,local_new_deletes);
}

__global__ static void update_triangles2(int32_t* triangleCounts, //!< per-edge triangle counts
                                         int32_t* triangleOffsets, //!< per-edge triangle offsets
                                         int32_t* removeFlag,
                                         int32_t* triangleBuffers1,
                                         int32_t* triangleBuffers2, //!< per-edge triangle buffer
                                         int32_t  numEdges
) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t sanity1, sanity2;
  for (int32_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    if (removeFlag[i]) { //must remove triangles associated with this edge
      for (int32_t iter = triangleOffsets[i];iter!=triangleOffsets[i+1];++iter) {
        int32_t e1=atomicExch(&triangleBuffers1[iter],-1);
        int32_t e2=atomicExch(&triangleBuffers2[iter],-1);

        // only remove the associated triangle in the edge that will not be remove in this round.
        if (e1!=-1&&e2!=-1) {
          if (!removeFlag[e1]) {
            for (int32_t j=triangleOffsets[e1];j!=triangleOffsets[e1+1];++j) {
              int32_t ea=atomicAdd(&triangleBuffers1[j],0u);
              int32_t eb=atomicAdd(&triangleBuffers2[j],0u);
              if ((ea==i&&eb==e2)||(ea==e2&&eb==i)) {
                // printf("Thread %d will remove triangle %d:%d from edge %d\n", idx+1, ea+1, eb+1, e1+1);
                sanity1 = atomicExch(&triangleBuffers1[j],-1);
                sanity2 = atomicExch(&triangleBuffers2[j],-1);
                // printf("Thread %d sanity1: %d, sanity2: %d\n", idx+1, sanity1, sanity2);
                if (sanity1!=-1 && sanity2!=-1) {
                  atomicAdd(&triangleCounts[e1], -1);
                }
              }
            }
          }
        
          if (!removeFlag[e2]) {
            for (int32_t j=triangleOffsets[e2];j!=triangleOffsets[e2+1];++j) {
              int32_t ea=atomicAdd(&triangleBuffers1[j],0u);
              int32_t eb=atomicAdd(&triangleBuffers2[j],0u);
              if ((ea==i&&eb==e1)||(ea==e1&&eb==i)) {
                // printf("Thread %d will remove triangle %d:%d from edge %d\n", idx, ea+1, eb+1, e2+1);
                sanity1 = atomicExch(&triangleBuffers1[j],-1);
                sanity2 = atomicExch(&triangleBuffers2[j],-1);
                // printf("Thread %d sanity1: %d, sanity2: %d\n", idx+1, sanity1, sanity2);
                if (sanity1!=-1 && sanity2!=-1) {
                  atomicAdd(&triangleCounts[e2], -1);
                }
              }
            }
          }
        }
      }
    triangleCounts[i]=0;
    }
  }
}

int main(int argc, char * argv[]) {
  std::string test_filename;	
  if (argv[1] == NULL) {
      test_filename = "./data/test2.bel";
  } else {
    test_filename = argv[1];
  }
  EdgeListFile test_file(test_filename);

  // get the total number of edges in the file.
  std::vector<EdgeTy<int32_t>> edges;
  int32_t size = getNumEdges(test_filename);
  std::cout << "Numbers of edges in the file : " << size << std::endl;

  // read the bel file into the EdgeListFile
  int32_t numEdge = test_file.get_edges(edges, size);
  std::cout << "Confirmed read edges: " << numEdge << std::endl;
  
  COO<int32_t> coo_test = COO<int32_t>::from_edges<std::vector<EdgeTy<int32_t>>::iterator>(edges.begin(), edges.end());
  COOView<int32_t> test_view = coo_test.view();
  
  int32_t numEdges = test_view.nnz();
  int32_t numRows = test_view.num_rows();
  // vector<int32_t> triangleCount(numEdges);  // keep track of the number of triangles for each edge
  // vector<vector<pair<int32_t, int32_t>>> triangleList(numEdges); // keep track of the triangle edges for each edge
  std::cout << "numEdges from nnz: " << numEdges << std::endl;

	int32_t* edgeSrc_device = nullptr;
  int32_t* edgeDst_device = nullptr;
	int32_t* rowPtr_device = nullptr;
	int32_t* triangleCount = nullptr;
  int32_t* triangleOffsets = nullptr;
  int32_t* triangleOffCounts = nullptr;
  int32_t* triangleBuffers1 = nullptr;
  int32_t* triangleBuffers2 = nullptr;
  int32_t* removeFlag = nullptr;
  int* edge_exists = nullptr;
  int* new_deletes = nullptr;
  int k=2;

	//allocate necessary memory
	cudaMalloc(&edgeSrc_device, numEdges*sizeof(int32_t));
	cudaMalloc(&edgeDst_device, numEdges*sizeof(int32_t));
	cudaMalloc(&rowPtr_device, (numRows+1)*sizeof(int32_t));
	cudaMallocManaged(&triangleCount, numEdges*sizeof(int32_t));
  cudaMallocManaged(&triangleOffsets, (numEdges+1)*sizeof(int32_t));
  cudaMallocManaged(&triangleOffCounts, (numEdges+1)*sizeof(int32_t));
  cudaMallocManaged(&removeFlag, numEdges*sizeof(int32_t));
  cudaMallocManaged(&edge_exists, sizeof(int));
  cudaMallocManaged(&new_deletes, sizeof(int));
  *edge_exists=1;
  *new_deletes=0;
	//copy over data
	cudaMemcpy(edgeSrc_device, test_view.row_ind(), numEdges*sizeof(int32_t),cudaMemcpyHostToDevice);
	cudaMemcpy(edgeDst_device, test_view.col_ind(), numEdges*sizeof(int32_t),cudaMemcpyHostToDevice);
  cudaMemcpy(rowPtr_device, test_view.row_ptr(), (numRows+1)*sizeof(int32_t),cudaMemcpyHostToDevice);
  
  thrust::device_ptr<int32_t> removeFlag_ptr(removeFlag);
  thrust::fill(removeFlag_ptr,removeFlag_ptr+numEdges,0);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //call triangle_count
  dim3 dimBlock(512);
  dim3 dimGrid (ceil(numEdges * 1.0 / dimBlock.x));
  cudaEventRecord(start);
	count_triangle<<<dimBlock, dimGrid>>>(triangleCount, edgeSrc_device, edgeDst_device, rowPtr_device, numEdges);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Triangle counting time: " << milliseconds << " ms" << std::endl;

  // std::cout << "Triangle Count" << std::endl;
	// int32_t totalCount = 0;
	// for (int32_t i = 0; i < numEdges; ++i) {
	// 	std::cout << triangleCount[i] << ' ';
	// 	totalCount += triangleCount[i];
	// }
  // std::cout << "totalCount: " << totalCount << std::endl;

  thrust::device_ptr<int32_t> triangleCount_ptr(triangleCount);
  triangleOffsets[0]=0;
  thrust::device_ptr<int32_t> triangleOffsets_ptr(triangleOffsets);
  thrust::inclusive_scan(triangleCount_ptr,triangleCount_ptr+numEdges,triangleOffsets_ptr+1);
  cudaDeviceSynchronize();
  thrust::device_ptr<int32_t> triangleOffCounts_ptr(triangleOffCounts);
  thrust::copy(triangleOffsets_ptr,triangleOffsets_ptr+numEdges+1,triangleOffCounts_ptr);

  // std::cout << "Scan:" << std::endl;
  // for (int32_t i = 0; i < numEdges+1; ++i) {
  // 	std::cout << triangleOffsets_ptr[i] << " ";
  // }
  // std::cout << std::endl;
  
  cudaMallocManaged(&triangleBuffers1,triangleOffsets[numEdges]*sizeof(int32_t));
  cudaMallocManaged(&triangleBuffers2,triangleOffsets[numEdges]*sizeof(int32_t));
  write_triangle<<<dimBlock, dimGrid>>>(triangleOffsets, triangleOffCounts, triangleBuffers1, triangleBuffers2, edgeSrc_device, edgeDst_device, rowPtr_device, numEdges);
  cudaDeviceSynchronize();


  // std::cout << "Triangle Write" << std::endl;
  // for (int32_t i = 0; i < triangleOffsets[numEdges]; ++i) {
  // 	std::cout << triangleBuffers1[i] << ":" << triangleBuffers2[i] << '\t';
  // }
  // std::cout << std::endl;
  // std::cout << "after update_triangles" << std::endl;
  //   for (int32_t i = 0; i < triangleOffsets[numEdges]; ++i) {
  //     if (!(triangleBuffers1[i]+1==0&&triangleBuffers2[i]+1==0))
  //   	std::cout << triangleBuffers1[i]+1 << ":" << triangleBuffers2[i]+1 << '\t';
  //   }
  //   std::cout << std::endl;
  cudaEventRecord(start);
  while (*edge_exists) {
    if (*new_deletes==0) {
      //
      std::cout<<"k="<<k<<" "<<"Triangles Left: "<<thrust::reduce(triangleCount_ptr,triangleCount_ptr+numEdges,0);
      std::cout << " Edgeleft: " << numEdges-thrust::reduce(removeFlag,removeFlag+numEdges,0) << std::endl;
      ++k;
    }

    // std::cout << "Print at the begnning of each iteration" << std::endl;
    // std::cout<<"k="<<k<<" "<<"Triangles Left: "<<thrust::reduce(triangleCount_ptr,triangleCount_ptr+numEdges,0)<<std::endl;
    // for (int32_t i=0; i<numEdges;++i) {
    //   std::cout << (i+1) << ":\t" << triangleCount[i] << "\t";
    //   for (int32_t j=triangleOffsets[i];j!=triangleOffsets[i+1];++j) {
    //     std::cout<<triangleBuffers1[j]+1<<":"<<triangleBuffers2[j]+1<<"\t";
    //   }
    //   std::cout<<std::endl;
    // }
    // std::cout << std::endl;

    *edge_exists=0;
    *new_deletes=0;
    
    truss_decompose<<<dimBlock, dimGrid>>>(triangleCount, triangleOffsets, removeFlag, triangleBuffers1, triangleBuffers2, edge_exists, new_deletes, numEdges, k);
    cudaDeviceSynchronize();

    // std::cout << "Triangle buffers" << std::endl;
    // for (int32_t i = 0; i < triangleOffsets[numEdges]; ++i) {
    // 	std::cout << triangleBuffers1[i] << ":" << triangleBuffers2[i] << '\t';
    // }
    // std::cout << std::endl;

    if (*new_deletes==1) {
      update_triangles2<<<dimBlock, dimGrid>>>(triangleCount, triangleOffsets, removeFlag, triangleBuffers1, triangleBuffers2, numEdges);
      cudaDeviceSynchronize();
    }

    // std::cout << "Print right after update_triangles" << std::endl;
    // std::cout<<"k="<<k<<" "<<"Triangles Left: "<<thrust::reduce(triangleCount_ptr,triangleCount_ptr+numEdges,0)<<std::endl;
    // for (int32_t i=0; i<numEdges;++i) {
    //   std::cout << (i+1) << ":\t" << triangleCount[i] << "\t";
    //   for (int32_t j=triangleOffsets[i];j!=triangleOffsets[i+1];++j) {
    //     std::cout<<triangleBuffers1[j]+1<<":"<<triangleBuffers2[j]+1<<"\t";
    //   }
    //   std::cout<<std::endl;
    // }
    // std::cout << std::endl;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "k truss time: " << milliseconds << " ms" << std::endl;
  // std::cout << "Triangle Count" << std::endl;
	// int32_t totalCount = 0;
	// for (int32_t i = 0; i < numEdges; ++i) {
	// 	std::cout << triangleCount[i] << ' ';
	// 	totalCount += triangleCount[i];
	// }
  // std::cout << "totalCount: " << totalCount << std::endl;

  std::cout << "kmax = " << k << std::endl;
  
  cudaFree(edgeSrc_device);
  cudaFree(edgeDst_device);
  cudaFree(rowPtr_device);
  cudaFree(triangleCount);
  cudaFree(triangleOffsets);
  return 0;
}
