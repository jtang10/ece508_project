#include <map>
#include <vector>
#include <list>
#include <algorithm>
#include "util.cpp"
#include "edge_list_file.hpp"
#include "coo-impl.hpp"
#include <chrono>

#define  DEBUG false

using namespace std;

inline pair<int, int> set_edge_pair(int u, int v) {
    return u < v ? make_pair(u, v) : make_pair(v, u);
}

// when originEdge is removed, update 
int update_triangle(vector<int> &triangleCount, 
                     vector<vector<pair<int, int>>> &triangleList,
                     int originEdge
) { 
    int numRemoved = 0;
    if (triangleCount[originEdge]) {
        for (auto it = triangleList[originEdge].begin(); it != triangleList[originEdge].end(); it++) {
            int edge1 = it->first;
            int edge2 = it->second;
            pair<int, int> pair1 = set_edge_pair(originEdge, edge2);
            pair<int, int> pair2 = set_edge_pair(originEdge, edge1);
            vector<pair<int, int>>* list1 = &triangleList[edge1];
            vector<pair<int, int>>* list2 = &triangleList[edge2];

            // search for the edge pair, remove it and decrement triangle count.
            list1->erase(remove(list1->begin(), list1->end(), pair1), list1->end());
            triangleCount[edge1]--;

            list2->erase(remove(list2->begin(), list2->end(), pair2), list2->end());
            triangleCount[edge2]--;
            numRemoved++;
        }

        // clear its own triangles at the last.
        triangleList[originEdge].clear();
        triangleCount[originEdge] = 0;
    }

    return numRemoved;
}

/* Given the COO format of a graph, return the number of triangles and
 * store the count for each edge
 */
void triangle_count(vector<int> &triangleCount, 
                    vector<vector<pair<int, int>>> &triangleList,
					int &totalCount,
                    const int* edgeSrc, 
                    const int* edgeDst,
                    const int* rowPtr,
                    int numEdges
) {
    int count = 0;
    int u, v, u_ptr, v_ptr, u_end, v_end, w1, w2;

    for (int i = 0; i < numEdges; i++) {
        u = edgeSrc[i];
        v = edgeDst[i];

        u_ptr = rowPtr[u];
        v_ptr = rowPtr[v];

        u_end = rowPtr[u+1];
        v_end = rowPtr[v+1];

        w1 = edgeDst[u_ptr];
        w2 = edgeDst[v_ptr];

        while ((u_ptr<u_end) && (v_ptr<v_end) && (u < v)) {
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

int truss_decomposition(vector<int> &triangleCount, 
                        vector<vector<pair<int, int>>> &triangleList,
                        const int* edgeSrc, 
                        const int* edgeDst,
                        const int* rowPtr,
                        int numEdges,
                        int totalCount
) { 
    int k = 3;
    int numRemoved = 0;
    int removeFlag [numEdges] = {0};  // 0 means present; 1 means will be removed; 2 means already removed

    while (numRemoved < numEdges) {
        for (int i = 0; i < numEdges; ++i) {
            // if the edge is not removed yet and has less triangles than k-2,
            // label the edge to be removed and update the triangleList and triangleCount later.
            if (removeFlag[i]==0 && triangleCount[i]<(k-2)) {
                removeFlag[i] = 1;
                numRemoved++;
            }
        }

        if (DEBUG) {
            cout << endl << "k=" << k << ": ";
            for (int i = 0; i < numEdges; ++i) {
                cout << removeFlag[i] << ' ';
            }
            cout << endl;

            cout << "Before" << endl;
            for (int i = 0; i < numEdges; i++) {
                cout << i+1 << ": " << triangleCount[i] << '\t'; 
                for (auto it = triangleList[i].begin(); it != triangleList[i].end(); ++it) {
                    cout << it->first+1 << ':' << it->second+1 << '\t';
                }
                cout << endl;
            }
            cout << "totalCount: " << totalCount << endl;
            cout << "k=" << k << "; " << "numRemoved: " << numRemoved << endl;
        }

        for (int i = 0; i < numEdges; ++i) {
            if (removeFlag[i] == 1) {	
                totalCount -= update_triangle(triangleCount, triangleList, i);
                removeFlag[i] = 2;
            }
        }

        if (DEBUG) {
            cout << "After" << endl;
            for (int i = 0; i < numEdges; i++) {
                cout << i+1 << ": " << triangleCount[i] << '\t'; 
                for (auto it = triangleList[i].begin(); it != triangleList[i].end(); ++it) {
                    cout << it->first+1 << ':' << it->second+1 << '\t';
                }
                cout << endl;
            }
            cout << "totalCount: " << totalCount << endl;
        }

        if (totalCount) {
            k++;
        } else {
            k--;
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
    vector<EdgeTy<int>> edges;
    int size = getNumEdges(test_filename);
    cout << "Numbers of edges in the file : " << size << endl;

    // read the bel file into the EdgeListFile
    int numEdge = test_file.get_edges(edges, size);
    cout << "Confirmed read edges: " << numEdge << endl;

    // for (auto it = edges.begin(); it != edges.end(); it++) {
    //     cout << it->first << " " << it->second << endl;
    // }

    COO<int> coo_test = COO<int>::from_edges<vector<EdgeTy<int>>::iterator>(edges.begin(), edges.end());
    COOView<int> test_view = coo_test.view();
    int numEdges = test_view.nnz();
    vector<int> triangleCount(numEdges);  // keep track of the number of triangles for each edge
    vector<vector<pair<int, int>>> triangleList(numEdges); // keep track of the triangle edges for each edge
    bool *remove;
	int totalCount = 0;

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    triangle_count(triangleCount,
                   triangleList,
				   totalCount,
                   test_view.row_ind(),
                   test_view.col_ind(),
                   test_view.row_ptr(),
                   numEdges);
    chrono::steady_clock::time_point end= chrono::steady_clock::now();
    cout << "Triangle count time = " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << " us" << std::endl;

	cout << "Total number of triangles: " << totalCount << endl;
    // cout << "Triangle Count" << endl;
    // for (int i = 0; i < numEdges; i++) {
    //     cout << i << ": " << triangleCount[i] << '\t'; 
    //     for (auto it = triangleList[i].begin(); it != triangleList[i].end(); ++it) {
    //         cout << it->first << ':' << it->second << '\t';
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    begin = chrono::steady_clock::now();
    int k = truss_decomposition(triangleCount,
                                triangleList,
                                test_view.row_ind(),
                                test_view.col_ind(),
                                test_view.row_ptr(),
                                numEdges,
                                totalCount);
    end= chrono::steady_clock::now();                    
    cout << "Truss decomposition time = " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << " us" << std::endl;

    // cout << "Triangle Count" << endl;
    // for (int i = 0; i < numEdges; i++) {
    //     cout << i << ": " << triangleCount[i] << '\t'; 
    //     for (auto it = triangleList[i].begin(); it != triangleList[i].end(); ++it) {
    //         cout << it->first << ':' << it->second << '\t';
    //     }
    //     cout << endl;
    // }
    // cout << endl;
    cout << "max k = " << k << endl;

    // cout << "Number of rows in the COO: " << coo_test.num_rows() << endl;
    // cout << "Number of non-zero rows in the COO: " << coo_test.nnz() << endl;
    // cout << "Number of nodes in the COO: " << coo_test.num_nodes() << endl;

    // cout << "COOView members: " << endl;
    // cout << "nnz: " << test_view.nnz() << endl;
    // cout << "num_rows: " << test_view.num_rows() << endl;

    // cout << "row_ptr:" << endl;
    // for (int i = 0; i <= test_view.num_rows(); ++i)
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
