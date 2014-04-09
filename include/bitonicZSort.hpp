#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "cli.hpp"
#include "kernels.hpp"
#include "stl.hpp"


using namespace std;

//#define DATA_TYPE int 
//#define DATA_SIZE 1024


class stlclBitonicZSort : public CLI
{
    public:


        stlclBitonicZSort(
                const char* kernalSource, 
                const char* kernalName) 
        : CLI ( kernalSource, kernalName) {}


};

