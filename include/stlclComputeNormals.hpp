
// compute STL normal vectors for
// given verticies on GPU
// 

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "cli.hpp"

using namespace std;

class stlclComputeNormals : public CLI
{

public:

    stlclComputeNormals(
        const char* kernalSource, 
        const char* kernalName) : CLI ( kernalSource, kernalName) 
    {}




};