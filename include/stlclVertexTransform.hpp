
// transform STL verticies on GPU
// notes:
//test in GLGraphicWidget.cpp
//glMultMatrixd(m_viewpoint.transformMatrix().data())


#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "cli.hpp"

#define VERTEX_TRANFORM_SIZE 12

using namespace std;

class stlclVertexTransform : public CLI
{
public:
    stlclVertexTransform(
        const char* kernalSource, 
        const char* kernalName) : CLI ( kernalSource, kernalName)
    {}

    cl_mem VertexTransform(
        float* transform, 
        std::vector<float> &verticies)
    {
        cl_int localstatus;
        unsigned int nVerticies = verticies.size();
        size_t vertexBytes = nVerticies * sizeof(float);
        float transformArray[VERTEX_TRANFORM_SIZE];
        
        printf("wat\n");
        for (int i = 0; i < VERTEX_TRANFORM_SIZE; ++i)
        {
            transformArray[i] = transform[i]; 
        }

        cl_mem bufferA = KernelArgs(
            transformArray,
            VERTEX_TRANFORM_SIZE*sizeof(float),
            0,
            CL_MEM_READ_ONLY);

        cl_mem bufferB = KernelArgs(
            &verticies.front(),
            vertexBytes,
            1,
            CL_MEM_READ_WRITE);

        cl_mem bufferC = clCreateBuffer(
            context, 
            CL_MEM_WRITE_ONLY, 
            vertexBytes, 
            NULL,
            &localstatus);

        clSetKernelArg(
            kernel, 
            2, 
            sizeof(cl_mem), 
            (void *) &bufferC); 

        errors.push_back(localstatus);
/*
        cl_mem bufferC = KernelArgs(
            vertexBuffer,
            vertexBytes,
            2,
            CL_MEM_WRITE_ONLY);
*/

        // Define an index space (global work size) of work 
        // items for 
        // execution. A workgroup size (local work size) is not 
        // required, 
        // but can be used.
        size_t globalWorkSize[1];
        
        // There are 'elements' work-items 
        globalWorkSize[0] = nVerticies/9;

        // STEP 11: Enqueue the kernel for execution
        
        // Execute the kernel by using 
        // clEnqueueNDRangeKernel().
        // 'globalWorkSize' is the 1D dimension of the 
        // work-items
        localstatus = clEnqueueNDRangeKernel(
            cmdQueue, 
            kernel, 
            1, 
            NULL, 
            globalWorkSize, 
            NULL, 
            0, 
            NULL, 
            NULL);

        errors.push_back(localstatus);


        // STEP 12: Read the output buffer back to the host
        
        /*
        clEnqueueReadBuffer(
            cmdQueue, 
            bufferC, 
            CL_TRUE,        // CL_TRUE is a BLOCKING read
            0, 
            vertexBytes, 
            vertexBuffer, 
            0, 
            NULL, 
            NULL);
        */

        // Free OpenCL resources
        clReleaseMemObject(bufferA);
        clReleaseMemObject(bufferB);
        //clReleaseMemObject(bufferC);

        // Free host resources

        return bufferC;
    }

};
