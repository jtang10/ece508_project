#include "coo.hpp"
#define NUM_BLOCKS 4
__device__ uint64_t triangles_buffer_offset=0;
__device__ void triangle_count(COOView<int> graph, int* triangles_count) {
	int* edgeDsts=graph.col_ind();
	int* rowPtrs=graph.row_ptr();
	int* edgeSrcs=graph.row_ind();
	uint64_t num_rows=graph.num_rows();
	uint64_t num_edges=graph.nnz();
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
__device__ void triangle_write(COOView<int> graph, int* triangles_buffer, int* triangles_offsets) {
	int* edgeDsts=graph.col_ind();
	int* rowPtrs=graph.row_ptr();
	int* edgeSrcs=graph.row_ind();
	uint64_t num_rows=graph.num_rows();
	uint64_t num_edges=graph.nnz();
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
					++triangle_offset;
					//add to set of triangles for this edge
				}
			}
		}
	}
}
__device__ void truss_decomposition(COOView<int> graph,int* triangle_counts, int* triangles_offsets, int* triangles_buffer,int* edge_exists, int* new_deletes, int k) {
	int* edgeDsts=graph.col_ind();
	int* rowPtrs=graph.row_ptr();
	int* edgeSrcs=graph.row_ind();
	uint64_t num_rows=graph.num_rows();
	uint64_t num_edges=graph.nnz();
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
		else {
			local_edge_exists=1;
		}
	}
	atomicCAS(edge_exists,0,local_edge_exists);
	atomicCAS(new_deletes,0,local_new_deletes);
}

void truss_wrapper(COOView<int> graph) {
	//allocate necessary memory
	//call triangle_count
	//allocate necessary memory
	//call triangle_write
	int k=2;
	int edge_exists=1;
	int new_deletes=0;
	while (edge_exists) {
		if (new_deletes==0) {
			//output current graph as k-truss subgraph
		}
		edge_exists=0;
		new_deletes=0;
		//call truss_decomposition here
	}
}
