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
#include <thrust/copy.h>
#include <utility>
#include <iostream> 

// row_ptr:
// 0 0 2 6 10 13 16 
// col_index:
// 2 3 1 3 4 5 1 2 4 5 2 3 5 2 3 4 
// row_index:
// 1 1 2 2 2 2 3 3 3 3 4 4 4 5 5 5 


__global__ static void count_triangle(int32_t *triangleCount,        //!< per-edge triangle counts
                                      const int32_t *const edgeSrc,  //!< node ids for edge srcs
                                      const int32_t *const edgeDst,  //!< node ids for edge dsts
                                      const int32_t *const rowPtr,   //!< source node offsets in edgeDst
                                      const int32_t numEdges         //!< how many edges to count triangles for
) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  for(int32_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    // Determine the source and destination node for the edge
    int32_t u = edgeSrc[i];
    int32_t v = edgeDst[i];

    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    int32_t u_ptr = rowPtr[u];
    int32_t v_ptr = rowPtr[v];

    int32_t u_end = rowPtr[u + 1];
    int32_t v_end = rowPtr[v + 1];

    int32_t w1 = edgeDst[u_ptr];
    int32_t w2 = edgeDst[v_ptr];
    // printf("idx %d = %d * %d + %d, u:%d, v:%d, u_ptr:%d, v_ptr:%d, u_end:%d, v_end:%d, w1:%d, w2:%d\n", i, blockIdx.x, blockDim.x, threadIdx.x, u, v, u_ptr, v_ptr, u_end, v_end, w1, w2);

    // Determine how many elements of those two arrays are common
    while ((u_ptr < u_end) && (v_ptr < v_end)) {
      if (w1 < w2) {
        u_ptr++;
        if (u_ptr < u_end) 
          w1 = edgeDst[u_ptr];
      } else if (w1 > w2) {
        v_ptr++;
        if (v_ptr < v_end) 
          w2 = edgeDst[v_ptr];
      } else {
        atomicAdd(triangleCount+i, 1);
        atomicAdd(triangleCount+u_ptr, 1);
        atomicAdd(triangleCount+v_ptr, 1);
        u_ptr++;
        v_ptr++;
        if (u_ptr < u_end) 
          w1 = edgeDst[u_ptr];
        if (v_ptr < v_end) 
          w2 = edgeDst[v_ptr];
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
        u_ptr++;
        if (u_ptr < u_end) 
          w1 = edgeDst[u_ptr];
      } else if (w1 > w2) {
        v_ptr++;
        if (v_ptr < v_end) 
          w2 = edgeDst[v_ptr];
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
        u_ptr++;
        v_ptr++;
        if (u_ptr < u_end) 
          w1 = edgeDst[u_ptr];
        if (v_ptr < v_end) 
          w2 = edgeDst[v_ptr];
      }
    }
  }
}

__global__ static void truss_decompose(int32_t* edgeRemove,
                                       int32_t* triangleCounts, //!< per-edge triangle counts
                                       int32_t* triangleOffsets, //!< per-edge triangle offsets
                                       int32_t* triangleRemove, //!< per-edge triangle removes
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
      edgeRemove[i]=1;
      atomicAdd(triangleRemove+i,my_triangle_count);
      for (int32_t iter = triangleOffsets[i];iter!=triangleOffsets[i+1];++iter) {
        if (triangleBuffers1[iter]!=-1 && triangleBuffers2[iter]!=-1) {
          atomicAdd(triangleRemove+triangleBuffers1[iter],1);
          atomicAdd(triangleRemove+triangleBuffers2[iter],1);
        }
      }
    }
    else if (my_triangle_count>(0)) {
      local_edge_exists=1;
      edgeRemove[i]=0;
    } 
    // else {
    //   edgeRemove[i] = 2;
    // }
  }
  atomicCAS(edge_exists,0,local_edge_exists);
  atomicCAS(new_deletes,0,local_new_deletes);
}

__global__ static void adjust_remove(int32_t* edgeRemove, 
                                     int32_t* triangleOffsets, 
                                     int32_t* triangleBuffers1, 
                                     int32_t* triangleBuffers2, 
                                     int32_t* triangleRemove,
                                     int32_t numEdges
) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int32_t i = idx; i<numEdges; i+=blockDim.x*gridDim.x) {
    if (!edgeRemove[i]) {
      for (int32_t j=triangleOffsets[i];j!=triangleOffsets[i+1];++j) {
        int32_t e1=triangleBuffers1[j];
        int32_t e2=triangleBuffers2[j];
        // printf("Thread %d, e1: %d, e2: %d\n", i, e1, e2);
        if (e1 != -1 && e2 != -1) {
          if (edgeRemove[e1]==1&&edgeRemove[e2]==1) {
            --triangleRemove[i];
          }
        }
      }
    }
  }

}

__global__ static void update_triangles(int32_t* triangleCounts, //!< per-edge triangle counts
                                        int32_t* triangleOffsets, //!< per-edge triangle offsets
                                        int32_t* triangleRemove, //!< per-edge triangle removes
                                        int32_t* triangleBuffers1,
                                        int32_t* triangleBuffers2, //!< per-edge triangle buffer
                                        int32_t  numEdges
) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int32_t i = idx; i < numEdges; i += blockDim.x * gridDim.x) {
    if (triangleCounts[i]<=triangleRemove[i]) { //must remove triangles associated with this edge
      for (int32_t iter = triangleOffsets[i];iter!=triangleOffsets[i+1];++iter) {
        //int32_t e1=atomicAdd(&triangleBuffers1[iter],0u);
        //int32_t e2=atomicAdd(&triangleBuffers2[iter],0u);
        int32_t e1=atomicExch(&triangleBuffers1[iter],-1);
        int32_t e2=atomicExch(&triangleBuffers2[iter],-1);
        //printf("%d %d %d\n",i,e1,e2);
        if (e1!=-1&&e2!=-1) {
          for (int32_t j=triangleOffsets[e1];j!=triangleOffsets[e1+1];++j) {
            int32_t ea=atomicAdd(&triangleBuffers1[j],0u);
            int32_t eb=atomicAdd(&triangleBuffers2[j],0u);
            if ((ea==i&&eb==e2)||(ea==e2&&eb==i)) {
              //printf("%d %d %d\n",e1,ea,eb);
              atomicExch(&triangleBuffers1[j],-1);
              atomicExch(&triangleBuffers2[j],-1);
            }
          }
          for (int32_t j=triangleOffsets[e2];j!=triangleOffsets[e2+1];++j) {
            int32_t ea=atomicAdd(&triangleBuffers1[j],0u);
            int32_t eb=atomicAdd(&triangleBuffers2[j],0u);
            if ((ea==i&&eb==e1)||(ea==e1&&eb==i)) {
              //printf("%d %d %d\n",e2,ea,eb);
              atomicExch(&triangleBuffers1[j],-1);
              atomicExch(&triangleBuffers2[j],-1);
            }
          }
        }
      }
      triangleCounts[i]=0;
    }
    else {
      triangleCounts[i]-=triangleRemove[i];
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
  // std::cout << "Numbers of edges in the file : " << size << std::endl;

  // read the bel file into the EdgeListFile
  int32_t numEdge = test_file.get_edges(edges, size);
  // std::cout << "Confirmed read edges: " << numEdge << std::endl;
  
  COO<int32_t> coo_test = COO<int32_t>::from_edges<std::vector<EdgeTy<int32_t>>::iterator>(edges.begin(), edges.end());
  COOView<int32_t> test_view = coo_test.view();
  
  int32_t numEdges = test_view.nnz();
  int32_t numRows = test_view.num_rows();
  // std::cout << "numEdges from nnz: " << numEdges << std::endl;

  cudaEvent_t start, stop, start_total, stop_total;
  float milliseconds = 0;
  cudaEventCreate(&start_total);
  cudaEventCreate(&stop_total);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start_total);

	int32_t* edgeSrc_device = nullptr;
  int32_t* edgeDst_device = nullptr;
	int32_t* rowPtr_device = nullptr;
	int32_t* triangleCount = nullptr;
  int32_t* triangleOffsets = nullptr;
  int32_t* triangleOffCounts = nullptr;
  int32_t* triangleBuffers1 = nullptr;
  int32_t* triangleBuffers2 = nullptr;
  int32_t* triangleRemove = nullptr;
  int32_t* edgeRemove = nullptr;
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
  
	//copy over data
	cudaMemcpy(edgeSrc_device, test_view.row_ind(), numEdges*sizeof(int32_t),cudaMemcpyHostToDevice);
	cudaMemcpy(edgeDst_device, test_view.col_ind(), numEdges*sizeof(int32_t),cudaMemcpyHostToDevice);
	cudaMemcpy(rowPtr_device, test_view.row_ptr(), (numRows+1)*sizeof(int32_t),cudaMemcpyHostToDevice);
  
  cudaEventRecord(start);
  
  //call triangle_count
  dim3 dimBlock(512);
  dim3 dimGrid (ceil(numEdges * 1.0 / dimBlock.x));
	count_triangle<<<dimGrid, dimBlock>>>(triangleCount, edgeSrc_device, edgeDst_device, rowPtr_device, numEdges);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "TC time: " << milliseconds << " ms" << std::endl;

  // std::cout << "Triangle Count" << std::endl;
	// int32_t totalCount = 0;
	// for (int32_t i = 0; i < numEdges; ++i) {
	// 	// std::cout << triangleCount[i] << ' ';
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
  
  cudaMallocManaged(&triangleBuffers1,triangleOffsets[numEdges]*sizeof(int32_t));
  cudaMallocManaged(&triangleBuffers2,triangleOffsets[numEdges]*sizeof(int32_t));
  write_triangle<<<dimGrid, dimBlock>>>(triangleOffsets, triangleOffCounts, triangleBuffers1, triangleBuffers2, edgeSrc_device, edgeDst_device, rowPtr_device, numEdges);
  cudaDeviceSynchronize();
  
  cudaFree(edgeSrc_device);
  cudaFree(edgeDst_device);
  cudaFree(rowPtr_device);
  cudaFree(triangleOffCounts);
  
  // std::cout << "TriangleOffsets" << std::endl;
  // for (int32_t i = 0; i < numEdges+1; ++i) {
  //   std::cout << triangleOffsets[i] << ' ';
  // }
  // std::cout << std::endl;

  // std::cout << "TriangleOffCounts" << std::endl;
  // for (int32_t i = 0; i < numEdges+1; ++i) {
  //   std::cout << triangleOffsets[i] << ' ';
  // }
  // std::cout << std::endl;

  // start recording time for k-truss
  cudaEventRecord(start);
  
  cudaMallocManaged(&edgeRemove, numEdges*sizeof(int32_t));
  cudaMallocManaged(&triangleRemove, numEdges*sizeof(int32_t));
  cudaMallocManaged(&edge_exists, sizeof(int));
  cudaMallocManaged(&new_deletes, sizeof(int));

  *edge_exists=1;
  *new_deletes=0;
  
  // initialize all edgeRemove to 0
  // thrust::device_ptr<int32_t> edgeRemove_ptr(edgeRemove);
  // thrust::fill(edgeRemove_ptr,edgeRemove_ptr+numEdges,0);
  // cudaDeviceSynchronize();
  int32_t roundRemove = 0;
  int32_t edgeLeft = 0;
  
  while (*edge_exists) {
    if (*new_deletes==0) {
      edgeLeft = 0;
      for (int i = 0; i < numEdges; ++i) {
        if (!edgeRemove[i]) {
          edgeLeft++;
        }
      }
      // std::cout << "k=" << k << '\t' << "iter=" << roundRemove << '\t' << "\tTrianglesLeft: " << thrust::reduce(triangleCount_ptr,triangleCount_ptr+numEdges,0) << "\tEdgesleft: " << edgeLeft << std::endl;
      ++k;

      // std::cout << "Triangle Write when k=" << k-1 << std::endl;
      // int32_t scan_count = 1;
      // std::cout << "0: " << triangleCount[0] << "\t";
      // for (int32_t i = 0; i < triangleOffsets[numEdges]; ++i) {
      //   if (i < triangleOffsets[scan_count]) {
      //     std::cout << triangleBuffers1[i] << ":" << triangleBuffers2[i] << '\t';
      //   } else {
      //     std::cout << std::endl << scan_count << ": " << triangleCount[scan_count] << "\t";
      //     scan_count++;
      //   }
      // }
      // std::cout << std::endl;
      
      roundRemove = 0;
    }
    *edge_exists=0;
    *new_deletes=0;
    roundRemove++;
    thrust::device_ptr<int32_t> triangleRemove_ptr(triangleRemove);
    thrust::fill(triangleRemove_ptr,triangleRemove_ptr+numEdges,0);
    cudaDeviceSynchronize();
    truss_decompose<<<dimGrid, dimBlock>>>(edgeRemove, triangleCount, triangleOffsets, triangleRemove, triangleBuffers1, triangleBuffers2, edge_exists, new_deletes, numEdges, k);
    cudaDeviceSynchronize();
    adjust_remove<<<dimGrid, dimBlock>>>(edgeRemove, triangleOffsets, triangleBuffers1, triangleBuffers2, triangleRemove,numEdges);
    cudaDeviceSynchronize();

    if (*new_deletes==1) {
      update_triangles<<<dimGrid, dimBlock>>>(triangleCount, triangleOffsets, triangleRemove, triangleBuffers1, triangleBuffers2, numEdges);
      //update_triangles_cpu(triangleCount, triangleOffsets, triangleRemove, triangleBuffers1, triangleBuffers2, numEdges);
    cudaDeviceSynchronize();
    }
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "k truss time: " << milliseconds << " ms" << std::endl;
  std::cout << "kmax = " << (k-1) << std::endl;

  cudaFree(triangleCount);
  cudaFree(triangleOffsets);
  cudaFree(triangleRemove);
  cudaFree(edge_exists);
  cudaFree(new_deletes);
  cudaFree(edgeRemove);
  cudaFree(triangleBuffers1);
  cudaFree(triangleBuffers2);

  cudaEventRecord(stop_total);
  cudaEventSynchronize(stop_total);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_total, stop_total);
  std::cout << "total time: " << milliseconds << " ms" << std::endl;
  
  return 0;
}
