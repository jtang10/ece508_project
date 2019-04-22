#include <vector>
#include "util.cpp"
#include "edge_list_file.hpp"
#include "coo-impl.hpp"

using namespace std;

/* Given the COO format of a graph, return the number of triangles and
 * store the count for each edge
 */
// int triangle_count(COO<int> coo) {

// }
int main() {
    string test_filename("./data/test1.bel");
    EdgeListFile test_file(test_filename);

    // get the total number of edges in the file.
    vector<EdgeTy<int>> edges;
    int size = getNumEdges(test_filename);
    cout << "Numbers of edges in the file : " << size << endl;

    // read the bel file into the EdgeListFile
    int numEdge = test_file.get_edges(edges, size);
    cout << "Confirmed read edges: " << numEdge << endl;

    for (auto it = edges.begin(); it != edges.end(); it++) {
        cout << it->first << " " << it->second << endl;
    }

    // COO<int> coo_test;
    // coo_test.from_edges<vector<EdgeTy<int>>::iterator>(edges.begin(), edges.end());
    COO<int> coo_test = COO<int>::from_edges<vector<EdgeTy<int>>::iterator>(edges.begin(), edges.end());
    cout << "Number of rows in the COO: " << coo_test.num_rows() << endl;
    cout << "Number of non-zero rows in the COO: " << coo_test.nnz() << endl;
    cout << "Number of nodes in the COO: " << coo_test.num_nodes() << endl;

    COOView<int> test_view = coo_test.view();
    cout << "COOView members: " << endl;
    cout << "nnz: " << test_view.nnz() << endl;
    cout << "num_rows: " << test_view.num_rows() << endl;

    cout << "row_ptr:" << endl;
    for (int i = 0; i < test_view.num_rows(); ++i)
        cout << *(test_view.row_ptr()+i) << ' '; 
    cout << endl;

    cout << "col_index:" << endl;
    for (auto it = coo_test.colInd_.begin(); it != coo_test.colInd_.end(); it++) 
        cout << *it << ' ';
    cout << endl;

    return 0;
}
