#include <iostream>
#include <fstream>

/*
 * Given the filename of bel file, get the total number of edges.
 * By the nature of bel file, the number of edges is the file size divided by 24.
 */
int getNumEdges(const std::string &fileName) {
    std::ifstream file(fileName.c_str(), std::ifstream::in | std::ifstream::binary);

    if(!file.is_open())
        return -1;

    file.seekg(0, std::ios::end);
    int fileSize = file.tellg();
    file.close();

    return fileSize/24;
}