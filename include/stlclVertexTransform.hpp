
// transform STL verticies on GPU
// notes:
//test in GLGraphicWidget.cpp
//glMultMatrixd(m_viewpoint.transformMatrix().data())


#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "cli.hpp"


using namespace std;

class stlclVertexTransform : public CLI
{
public:
    stlclVertexTransform(
        const char* kernalSource, 
        const char* kernalName) : CLI ( kernalSource, kernalName)
    {}

    

};
