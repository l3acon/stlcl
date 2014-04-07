
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

void ComputeNormals(
    unsigned int nVerticies,
    float *verticies, 
    float *normalBuffer) 
{
    cl_int localstatus;

    //size_t vertexBytes = sizeof(float)*12;
    size_t vertexBytes = nVerticies * sizeof(float);
    size_t normalBytes = (nVerticies * sizeof(float))/3;

    //initalize CL interface and build kernel
    // declare CL memory buffers
    cl_mem bufferA = KernelArgs(
        verticies,
        vertexBytes,
        0,
        CL_MEM_READ_ONLY);


    cl_mem bufferC = KernelArgs(
        normalBuffer,
        normalBytes,
        1,
        CL_MEM_WRITE_ONLY);

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

    clEnqueueReadBuffer(
        cmdQueue, 
        bufferC, 
        CL_TRUE,            // CL_TRUE is a BLOCKING read
        0, 
        normalBytes, 
        normalBuffer, 
        0, 
        NULL, 
        NULL);


    // Free OpenCL resources
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferC);

    // Free host resources

    return ;
}

void ComputeNormal(
    unsigned int nVerticies,
    cl_mem verticies, 
    float *normalBuffer) 
{
    cl_int localstatus;

    //size_t vertexBytes = sizeof(float)*12;
    //size_t vertexBytes = nVerticies * sizeof(float);
    size_t normalBytes = (nVerticies * sizeof(float))/3;

    //initalize CL interface and build kernel
    // declare CL memory buffers

    clSetKernelArg(
        kernel, 
        0, 
        sizeof(cl_mem), 
        verticies);


    cl_mem buffer = KernelArgs(
        normalBuffer,
        normalBytes,
        1,
        CL_MEM_WRITE_ONLY);

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

    clEnqueueReadBuffer(
        cmdQueue, 
        buffer, 
        CL_TRUE,            // CL_TRUE is a BLOCKING read
        0, 
        normalBytes, 
        normalBuffer, 
        0, 
        NULL, 
        NULL);

    // Free OpenCL resources
    clReleaseMemObject(buffer);

    // Free host resources

    return ;
}


};