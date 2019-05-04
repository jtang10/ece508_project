#include <map>
#include <vector>
#include <list>
#include <cassert>
#include <algorithm>
#include "util.cpp"
#include "edge_list_file.hpp"
#include "coo-impl.hpp"
#include <chrono>
#include <numeric>

#define  DEBUG false

using namespace std;

inline pair<uint64_t, uint64_t> set_edge_pair(uint64_t u, uint64_t v) {
    return u < v ? make_pair(u, v) : make_pair(v, u);
}

// when originEdge is removed, iterate through every triangle it involves, delete that triangle record in the other 
// two edges and decrement the triangleCount.
uint64_t update_triangle(vector<uint64_t> &triangleCount, 
                    vector<vector<pair<uint64_t, uint64_t>>> &triangleList,
                    uint64_t originEdge
) { 
    uint64_t numRemoved = 0;
    for (auto it = triangleList[originEdge].begin(); it != triangleList[originEdge].end(); it++) {
        uint64_t edge1 = it->first;
        uint64_t edge2 = it->second;
        pair<uint64_t, uint64_t> pair1 = set_edge_pair(originEdge, edge2);
        pair<uint64_t, uint64_t> pair2 = set_edge_pair(originEdge, edge1);
        vector<pair<uint64_t, uint64_t>>* list1 = &triangleList[edge1];
        vector<pair<uint64_t, uint64_t>>* list2 = &triangleList[edge2];

        // search for the edge pair, remove it and decrement triangle coun t.
        list1->erase(remove(list1->begin(), list1->end(), pair1), list1->end());
        assert(list1->size() == (triangleCount[edge1]-1) && "the triangle is not successfully erased from list1");
        triangleCount[edge1]--;

        list2->erase(remove(list2->begin(), list2->end(), pair2), list2->end());
        assert(list2->size() == (triangleCount[edge2]-1) && "the triangle is not successfully erased from list2");
        triangleCount[edge2]--;
    }

    // clear its own triangles at the last.
    triangleList[originEdge].clear();
    numRemoved += triangleCount[originEdge]*3;
    triangleCount[originEdge] = 0;

    return numRemoved;
}

/* Given the COO format of a graph, return the number of triangles and
 * store the count for each edge
 */
void triangle_count(vector<uint64_t> &triangleCount, 
                    vector<vector<pair<uint64_t, uint64_t>>> &triangleList,
					uint64_t &totalCount,
                    const uint64_t* edgeSrc, 
                    const uint64_t* edgeDst,
                    const uint64_t* rowPtr,
                    uint64_t numEdges
) {
    uint64_t u, v, u_ptr, v_ptr, u_end, v_end, w1, w2;

    for (uint64_t i = 0; i < numEdges; i++) {
        u = edgeSrc[i];
        v = edgeDst[i];

        u_ptr = rowPtr[u];
        v_ptr = rowPtr[v];

        u_end = rowPtr[u+1];
        v_end = rowPtr[v+1];

        w1 = edgeDst[u_ptr];
        w2 = edgeDst[v_ptr];

        while ((u_ptr<u_end) && (v_ptr<v_end)) {
            if (w1 < w2) {
                w1 = edgeDst[++u_ptr];
            } else if (w1 > w2) {
                w2 = edgeDst[++v_ptr];
            } else {
                triangleCount[i]++;
				totalCount++;
                triangleCount[u_ptr]++;
                triangleCount[v_ptr]++;
                triangleList[i].push_back(set_edge_pair(u_ptr, v_ptr));
                triangleList[u_ptr].push_back(set_edge_pair(i, v_ptr));
                triangleList[v_ptr].push_back(set_edge_pair(i, u_ptr));
                w1 = edgeDst[++u_ptr];
                w2 = edgeDst[++v_ptr];
            }
        }
    }
}

uint64_t truss_decomposition2(vector<uint64_t> triangleCount, 
							  vector<vector<pair<uint64_t, uint64_t>>> triangleList,
							  const uint64_t* edgeSrc, 
							  const uint64_t* edgeDst,
							  const uint64_t* rowPtr,
							  uint64_t numEdges
) {
    uint64_t numtri=std::accumulate(triangleCount.begin(),triangleCount.end(),0);
	bool edgeExists = true;
	bool newDeletes = false;		
	uint64_t k = 2;
	uint64_t roundRemove = 0;	
	uint64_t edgeRemoved = 0;
    int removeFlag[numEdges] = {0};
    uint64_t triangleRemoved = 0;

	while (edgeExists) {
        if (DEBUG) {
            cout << "k=" << k << endl;
            for (uint64_t i = 0; i < numEdges; i++) {
                cout << i+1 << ": " << triangleCount[i] << '\t'; 
                for (auto it = triangleList[i].begin(); it != triangleList[i].end(); ++it) {
                    cout << it->first+1 << ':' << it->second+1 << '\t';
                }
                cout << endl;
            }
            cout << "edgeRemoved: " << edgeRemoved << endl;
        }

		if (!newDeletes) {
            cout << "k=" << k << ' ' << "iter=" << roundRemove << ' ' << "Edgesleft: " << numEdges-edgeRemoved << ' ' << "TrianglesLeft: " << numtri-triangleRemoved << endl;
			k++;
			roundRemove = 0;
		}
		edgeExists = false;
		newDeletes = false;
		roundRemove++;
	
		for (uint64_t i = 0; i < numEdges; ++i) {
            if (triangleCount[i] == 0 && removeFlag[i]==0) {
                edgeRemoved++;
                removeFlag[i] = 1;
			} else if (triangleCount[i]>0 && triangleCount[i]<(k-2)*3) {
                triangleRemoved += update_triangle(triangleCount, triangleList, i);
				newDeletes = true;
				edgeRemoved++;
                removeFlag[i] = 1;
			} else if (triangleCount[i] >= (k-2)*3) {
				edgeExists = true;
			}
		}
        
        // if (!edgeExists) 
	}
	
	return k;
}

uint64_t truss_decomposition(vector<uint64_t> triangleCount, 
                        vector<vector<pair<uint64_t, uint64_t>>> triangleList,
                        const uint64_t* edgeSrc, 
                        const uint64_t* edgeDst,
                        const uint64_t* rowPtr,
                        uint64_t numEdges
) { 
    uint64_t numtri=std::accumulate(triangleCount.begin(),triangleCount.end(),0);
    uint64_t k = 3;
    uint64_t edgeRemoved = 0;
    uint64_t roundRemove;
    uint64_t removeFlag [numEdges] = {0};  // 0 means present; 1 means will be removed; 2 means already removed
    bool flag = true;
    uint64_t totalCount = 0;
    uint64_t triangleRemoved = 0;

    while (edgeRemoved < numEdges) {
        roundRemove = 0;
        if (DEBUG) {
            cout << "Before" << endl;
            cout << endl << "k=" << k << ": " << endl << "removeFlag: ";
            for (uint64_t i = 0; i < numEdges; ++i) {
                cout << removeFlag[i] << ' ';
            }
            cout << endl;

            for (uint64_t i = 0; i < numEdges; i++) {
                cout << i+1 << ": " << triangleCount[i] << '\t'; 
                for (auto it = triangleList[i].begin(); it != triangleList[i].end(); ++it) {
                    cout << it->first+1 << ':' << it->second+1 << '\t';
                }
                cout << endl;
            }
        }
        
        while (flag) {
            roundRemove++;
            // edgeRemoved = 0;
            flag = false;
            for (uint64_t i = 0; i < numEdges; ++i) {
                // if the edge is not removed yet and has less triangles than k-2,
                // label the edge to be removed and update the triangleList and triangleCount later.
                if (removeFlag[i]==0 && triangleCount[i]<(k-2)*3) {
                    // if the edge has no triangle, just flag it as already removed.
                    if (triangleCount[i]) {
                        triangleRemoved += update_triangle(triangleCount, triangleList, i);
                        flag = true;
                    } 
                    removeFlag[i] = 1;
                    edgeRemoved++;
                }
            }

            if (DEBUG) {
                cout << "After " << roundRemove << endl << "removeFlag: ";
                for (uint64_t i = 0; i < numEdges; ++i) {
                    cout << removeFlag[i] << ' ';
                }
                cout << endl;
                for (uint64_t i = 0; i < numEdges; i++) {
                    cout << i+1 << ": " << triangleCount[i] << '\t'; 
                    for (auto it = triangleList[i].begin(); it != triangleList[i].end(); ++it) {
                        cout << it->first+1 << ':' << it->second+1 << '\t';
                    }
                    cout << endl;
                }
                cout << "edgeRemoved: " << edgeRemoved << endl;
            }
        }

        if (edgeRemoved < numEdges) {
            cout << "k=" << k << ' ' << "iter=" << roundRemove << ' ' << "Edgesleft: " << numEdges - edgeRemoved << " " << "TrianglesLeft: " << numtri-triangleRemoved << endl;
            k++;
            flag = true;
        } else {
            k--;  // everything is removed in this round, this k won't count so decrement
            break;
        }
    }

    return k;
}

int main(int argc, char * argv[]) {
 	string test_filename;	
	if (argv[1] == NULL) {
    	test_filename = "./data/test2.bel";
	} else {
		test_filename = argv[1];
	}
    EdgeListFile test_file(test_filename);

    // get the total number of edges in the file.
    vector<EdgeTy<uint64_t>> edges;
    uint64_t size = getNumEdges(test_filename);
    cout << "Numbers of edges in the file : " << size << endl;

    // read the bel file into the EdgeListFile
    uint64_t numEdge = test_file.get_edges(edges, size);
    cout << "Confirmed read edges: " << numEdge << endl;

    // for (auto it = edges.begin(); it != edges.end(); it++) {
    //     cout << it->first << " " << it->second << endl;
    // }

    COO<uint64_t> coo_test = COO<uint64_t>::from_edges<vector<EdgeTy<uint64_t>>::iterator>(edges.begin(), edges.end());
    COOView<uint64_t> test_view = coo_test.view();
    uint64_t numEdges = test_view.nnz();
    vector<uint64_t> triangleCount(numEdges);  // keep track of the number of triangles for each edge
    vector<vector<pair<uint64_t, uint64_t>>> triangleList(numEdges); // keep track of the triangle edges for each edge
	uint64_t totalCount = 0;
    cout << "numEdges from nnz: " << numEdges << endl;

    // uint64_t print_edges = numEdges < 20 ? numEdges : 20;
    // cout << "Row Indices" << endl;
    // for (uint64_t i = 0; i < print_edges; i++) {
    //     cout << test_view.row_ind()[i] << ' ';
    // }
    // cout << endl;

    // cout << "Col Indices" << endl;
    // for (uint64_t i = 0; i < print_edges; i++) {
    //     cout << test_view.col_ind()[i] << ' ';
    // }
    // cout << endl;

    // uint64_t print_row = (test_view.num_rows()+1) < 20 ? test_view.num_rows()+1 : 20;
    // cout << "numRows: " << test_view.num_rows() << endl;
    // cout << "Row Pointer" << endl;
    // for (uint64_t i = 0; i < print_row; i++) {
    //     cout << test_view.row_ptr()[i] << ' ';
    // }
    // cout << endl;

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    triangle_count(triangleCount, triangleList, totalCount, test_view.row_ind(), test_view.col_ind(), test_view.row_ptr(), numEdges);
    chrono::steady_clock::time_point end= chrono::steady_clock::now();
    cout << "Triangle count time = " << chrono::duration_cast<chrono::milliseconds> (end - begin).count() << " ms" << std::endl;
	cout << "Total number of triangles: " << totalCount << endl;

    // cout << "Triangle Count" << endl;
    // for (uint64_t i = 0; i < numEdges; i++) {
    //     cout << i+1 << ": " << triangleCount[i] << '\t'; 
    //     for (auto it = triangleList[i].begin(); it != triangleList[i].end(); ++it) {
    //         cout << it->first+1 << ':' << it->second+1 << '\t';
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    // begin = chrono::steady_clock::now();
    // uint64_t k = truss_decomposition(triangleCount, triangleList, test_view.row_ind(), test_view.col_ind(), test_view.row_ptr(), numEdges);
    // end= chrono::steady_clock::now();                    
    // cout << "Truss decomposition time = " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << " us" << std::endl;
    //cout << "max k = " << k << endl;

    begin = chrono::steady_clock::now();
    uint64_t k = truss_decomposition2(triangleCount, triangleList, test_view.row_ind(), test_view.col_ind(), test_view.row_ptr(), numEdges);
    end= chrono::steady_clock::now();                    
    cout << "Truss decomposition2 time = " << chrono::duration_cast<chrono::milliseconds> (end - begin).count() << " ms" << std::endl;
    cout << "max k = " << k << endl;

    // cout << "Triangle Count" << endl;
    // for (uint64_t i = 0; i < numEdges; i++) {
    //     cout << i << ": " << triangleCount[i] << '\t'; 
    //     for (auto it = triangleList[i].begin(); it != triangleList[i].end(); ++it) {
    //         cout << it->first << ':' << it->second << '\t';
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    // cout << "Number of rows in the COO: " << coo_test.num_rows() << endl;
    // cout << "Number of non-zero rows in the COO: " << coo_test.nnz() << endl;
    // cout << "Number of nodes in the COO: " << coo_test.num_nodes() << endl;

    // cout << "COOView members: " << endl;
    // cout << "nnz: " << test_view.nnz() << endl;
    // cout << "num_rows: " << test_view.num_rows() << endl;

    // cout << "row_ptr:" << endl;
    // for (uint64_t i = 0; i <= test_view.num_rows(); ++i)
    //     cout << *(test_view.row_ptr()+i) << ' '; 
    // cout << endl;

    // cout << "col_index:" << endl;
    // for (auto it = coo_test.colInd_.begin(); it != coo_test.colInd_.end(); it++) 
    //     cout << *it << ' ';
    // cout << endl;

    // cout << "row_index:" << endl;
    // for (auto it = coo_test.rowInd_.begin(); it != coo_test.rowInd_.end(); it++) 
    //     cout << *it << ' ';
    // cout << endl;

    return 0;
}
