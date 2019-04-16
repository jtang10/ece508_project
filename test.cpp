#include <vector>
#include <iostream>
#include <fstream>
#include "edge_list_file.hpp"

using namespace std;

int getNumEdges(const string &fileName)
{
    ifstream file(fileName.c_str(), ifstream::in | ifstream::binary);

    if(!file.is_open())
    {
        return -1;
    }

    file.seekg(0, ios::end);
    int fileSize = file.tellg();
    file.close();

    return fileSize/24;
}

int main() {
    EdgeListFile test_file("./data/Theory-3-4-5-9-Bk.bel");
    vector<EdgeTy<size_t>> edges;
    size_t numEdge = test_file.get_edges(edges, 10);

    int size = getNumEdges("./data/Theory-3-4-5-9-Bk.bel");

    cout << "Numbers of lines in the file : " << size << endl;

    cout << numEdge << endl;

    for (int i = 0; i < 10; ++i) {
        cout << edges.at(i).first << ", " << edges.at(i).second << endl;
    }

    return 0;
}
