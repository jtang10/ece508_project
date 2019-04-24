#include "coo-impl.hpp"
#include <iostream>
#include "modifiedfilereader.cpp"
#define NUM_BLOCKS 4
__device__ uint64_t triangles_buffer_offset=0;
__global__ void triangle_count(int* edgeDsts, int* rowPtrs, int* edgeSrcs, uint64_t num_rows, uint64_t num_edges, int* triangles_count) {
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	for (int i=tid;i<num_edges;i+=blockDim.x*NUM_BLOCKS) {
		int i_src=0;
		int i_dst=0;
		int srcSize=rowPtrs[edgeSrcs[i]+1]-rowPtrs[edgeSrcs[i]];
		int destSize=rowPtrs[edgeDsts[i]+1]-rowPtrs[edgeDsts[i]];
		if (edgeSrcs[i]>edgeDsts[i]) {
			int tricount=0;
			while (i_src<srcSize&&i_dst<destSize) {
				if (edgeDsts[rowPtrs[edgeSrcs[i]]+i_src]<edgeDsts[rowPtrs[edgeDsts[i]]+i_dst]) {
					++i_src;
				}
				else if (edgeDsts[rowPtrs[edgeSrcs[i]]+i_src]>edgeDsts[rowPtrs[edgeDsts[i]]+i_dst]) {
					++i_dst;
				}
				else {
					++i_src;
					++i_dst;
					++tricount;
				}
			}
			triangles_count[i]=tricount;
		}
	}
}
__global__ void triangle_write(int* edgeDsts, int* rowPtrs, int* edgeSrcs, uint64_t num_rows, uint64_t num_edges, int* triangles_buffer, int* triangles_offsets) {
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	for (int i=tid;i<num_edges;i+=blockDim.x*NUM_BLOCKS) {
		int i_src=0;
		int i_dst=0;
		int srcSize=rowPtrs[edgeSrcs[i]+1]-rowPtrs[edgeSrcs[i]];
		int destSize=rowPtrs[edgeDsts[i]+1]-rowPtrs[edgeDsts[i]];
		int triangle_offset=triangles_offsets[i];
		if (edgeSrcs[i]>edgeDsts[i]) {
			while (i_src<srcSize&&i_dst<destSize) {
				if (edgeDsts[rowPtrs[edgeSrcs[i]]+i_src]<edgeDsts[rowPtrs[edgeDsts[i]]+i_dst]) {
					++i_src;
				}
				else if (edgeDsts[rowPtrs[edgeSrcs[i]]+i_src]>edgeDsts[rowPtrs[edgeDsts[i]]+i_dst]) {
					++i_dst;
				}
				else {
					++i_src;
					++i_dst;
					triangles_buffer[triangle_offset]=rowPtrs[edgeSrcs[i]]+i_src;
					triangles_buffer[triangle_offset+1]=rowPtrs[edgeDsts[i]]+i_dst;
					triangle_offset+=2;
				}
			}
		}
	}
}
__global__ void truss_decomposition(int* edgeDsts, int* rowPtrs, int* edgeSrcs, uint64_t num_rows, uint64_t num_edges,int* triangles_counts, int* triangles_offsets, int* triangles_buffer,int* edge_exists, int* new_deletes, int k) {
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int local_edge_exists=0, local_new_deletes=0;
	for (int i=tid;i<num_edges;i+=blockDim.x*NUM_BLOCKS) {
		int tricount=atomicAdd(triangles_counts+i,0);
		if (tricount<(k-2)&&tricount>0) {
			triangles_counts[i]=0;
			local_new_deletes=1;
			for (int j=triangles_offsets[i];j!=triangles_offsets[i+1];j+=2) {
				int e1=triangles_buffer[j];
				int e2=triangles_buffer[j+1];
				for (int iter=triangles_offsets[e1];iter!=triangles_offsets[e1+1];iter+=2) {
					int ea=atomicAdd(triangles_buffer+iter,0);
					int eb=atomicAdd(triangles_buffer+iter+1,0);
					if (ea==i||eb==i) {
						atomicCAS(triangles_buffer+iter,ea,-1);
						atomicCAS(triangles_buffer+iter,eb,-1);
					}
				}
				for (int iter=triangles_offsets[e2];iter!=triangles_offsets[e2+1];iter+=2) {
					int ea=atomicAdd(triangles_buffer+iter,0);
					int eb=atomicAdd(triangles_buffer+iter+1,0);
					if (ea==i||eb==i) {
						atomicCAS(triangles_buffer+iter,ea,-1);
						atomicCAS(triangles_buffer+iter,eb,-1);
					}
				}
				atomicAdd(&triangles_counts[e1],-1);
				atomicAdd(&triangles_counts[e2],-1);
			}
		}
		else if(tricount>0) {
			local_edge_exists=1;
		}
	}
	atomicCAS(edge_exists,0,local_edge_exists);
	atomicCAS(new_deletes,0,local_new_deletes);
}

void truss_wrapper(COOView<int> graph) {
	int numThreadsPerBlock=128;
	const int* edgeDsts=graph.col_ind();
	const int* rowPtrs=graph.row_ptr();
	const int* edgeSrcs=graph.row_ind();
	uint64_t num_rows=graph.num_rows();
	uint64_t num_edges=graph.nnz();
	int* edgeDsts_d=nullptr;
	int* rowPtrs_d=nullptr;
	int* edgeSrcs_d=nullptr;
	int* triangles_count=nullptr;
	//allocate necessary memory
	cudaMalloc(&edgeDsts_d,num_edges*sizeof(int));
	cudaMalloc(&rowPtrs_d,(num_rows+1)*sizeof(int));
	cudaMalloc(&edgeSrcs_d,num_edges*sizeof(int));
	cudaMallocManaged(&triangles_count,num_edges*sizeof(int));
	//copy over data
	cudaMemcpy(edgeDsts_d, edgeDsts, num_edges*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(rowPtrs_d, rowPtrs, (num_rows+1)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(edgeSrcs_d, edgeSrcs, num_edges*sizeof(int),cudaMemcpyHostToDevice);
	//call triangle_count
	triangle_count<<<NUM_BLOCKS, numThreadsPerBlock>>>(edgeDsts_d,rowPtrs_d,edgeSrcs_d,num_rows,num_edges,triangles_count);
	cudaDeviceSynchronize();
	//allocate necessary memory
	int* triangles_buffer=nullptr;
	int* triangles_offsets=nullptr;
	cudaMallocManaged(&triangles_offsets,(num_edges+1)*sizeof(int));
	int start=0;
	for (int i=0;i!=num_edges;++i) {
		triangles_offsets[i]=start;
		start+=2*triangles_count[i];
	}
	triangles_offsets[num_edges]=start;
	cudaMallocManaged(&triangles_buffer,(triangles_offsets[num_edges])*sizeof(int));
	//call triangle_write
	triangle_write<<<NUM_BLOCKS, numThreadsPerBlock>>>(edgeDsts_d,rowPtrs_d, edgeSrcs_d, num_rows, num_edges, triangles_buffer, triangles_offsets);
	cudaDeviceSynchronize();
	int* edge_exists_ptr=nullptr;
	int* new_deletes_ptr=nullptr;
	int k=2;
	cudaMallocManaged(&edge_exists_ptr,sizeof(int));
	cudaMallocManaged(&new_deletes_ptr,sizeof(int));
	*edge_exists_ptr=1;
	*new_deletes_ptr=0;
	while (*edge_exists_ptr) {
		if (*new_deletes_ptr==0) {
			//output current graph as k-truss subgraph
			++k;
		}
		*edge_exists_ptr=0;
		*new_deletes_ptr=0;
		truss_decomposition<<<NUM_BLOCKS, numThreadsPerBlock>>>(edgeDsts_d, rowPtrs_d, edgeSrcs_d, num_rows, num_edges, triangles_count, triangles_offsets, triangles_buffer, edge_exists_ptr, new_deletes_ptr, k);
		cudaDeviceSynchronize();
	}
	cudaFree(edgeDsts_d);
	cudaFree(rowPtrs_d);
	cudaFree(edgeSrcs_d);
	cudaFree(triangles_count);
	cudaFree(triangles_offsets);
	cudaFree(triangles_buffer);
	cudaFree(edge_exists_ptr);
	cudaFree(new_deletes_ptr);
}
int main() {
	std::vector<std::pair<int,int>> edgetemp;
	EdgeListFile elf("Test.bel");
	elf.get_edges(edgetemp,4);
	COO<int> graph=COO<int>::from_edges(edgetemp.begin(),edgetemp.end());
	truss_wrapper(graph.view());
}