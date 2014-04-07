
// MATTHEW FERNANDEZ 2014
// 
// should probably add a license
// 
// benchmark/test program for STL transform on large STL files
// OpenCL implimentation for GPGPU
//

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <time.h>

#include "stl.hpp"

// the following inclues might be consolidated
#include "cli.hpp"
#include "bitonicZSort.hpp"
#include "stlclVertexTransform.hpp"
#include "stlclComputeNormals.hpp"
#include "kernels.hpp"


#define CL_ERRORS 1

#ifndef _WIN32
#ifndef __APPLE__
#define TIME 1
#define BENCHSIZE 10
#endif
#endif

int main() 
{
    const char* stlFile = "Ring.stl";

    std::vector<float> verticies;
    std::vector<float> normals;
    
    //later we can just use the memory in a std::vector?

    float A[12];

    //initalize our transform matrix naively
    for (int i = 0; i < 12; ++i)
        A[i] = (float) i;

    //file stuff
    if(stlRead(stlFile, verticies, normals))
    {
        std::cout<<"ERROR: reading file"<<std::endl;
        return 1;
    }

    //check sanity for verticies and normals
    if( fmod(verticies.size(),9.0) !=  0 || fmod(normals.size(),3.0) != 0 )
    {
        std::cout<<"ERROR: verticies and normals don't make sense up"<<std::endl;
        return 1;
    }


    // set up CLInterface resrouces
    stlclVertexTransform cliVertexTransform(
        stl_cl_vertexTransform_kernel_source, 
        "_kVertexTransform");
    cliVertexTransform.Initialize();
    cliVertexTransform.Build();
    
    stlclBitonicZSort cliBitonicZSort(
        bitonic_STL_sort_source,
        "_kBitonic_STL_Sort");
    cliBitonicZSort.Initialize();
    cliBitonicZSort.Build();

    stlclComputeNormals cliComputeNormals(
        stl_cl_computeNormals_kernel_source, 
        "_kComputeNormals");
    cliComputeNormals.Initialize();
    cliComputeNormals.Build();


    // do the benchmark
    #if TIME
    timespec watch[BENCHSIZE], stop[BENCHSIZE];
    for (int i = 0; i < BENCHSIZE; ++i)
    {
        clock_gettime(CLOCK_REALTIME, &watch[i]);
    #endif        
        
        // CPU Z sort
        //qsort(vertexBuffer, verticies.size()/9, sizeof(float)*9, vertex_comparator);

        cliBitonicZSort.TwosPad(verticies);
        float* vertexBuffer = (float*) malloc(sizeof(float) * verticies.size());
        float* normalBuffer = (float*) malloc(sizeof(float) * verticies.size()/3);

        cl_mem vertex_des = cliVertexTransform.VertexTransform( A, verticies);
        printf("VT done\n");
        cl_mem sort_des = cliBitonicZSort.Sort(
            vertexBuffer,
            CL_TRUE,       // blocking ?
            vertex_des);
        printf("Sort done\n");
        cl_mem cldes = cliComputeNormals.ComputeNormals(
            verticies.size(),
            normalBuffer,
            CL_TRUE,       // non-blocking?
            sort_des);
        printf("CN done\n");

        cliBitonicZSort.Wait();

        //clReleaseMemObject(cldes);
        


        #if CL_ERRORS
        printf("_kVertexTransform:\n");
        cliVertexTransform.PrintErrors();
        printf("_kbitonic_STL_Sort:\n");
        cliBitonicZSort.PrintErrors();
        printf("_kComputeNormal:\n");
        cliComputeNormals.PrintErrors();
        #endif
        
        free(vertexBuffer);
        free(normalBuffer);
    #if TIME    
        clock_gettime(CLOCK_REALTIME, &stop[i]); // Works on Linux but not OSX
    }

    double acc = 0.0;
    for (int i = 0; i < BENCHSIZE; ++i)
        acc += stop[i].tv_sec - watch[i].tv_sec + (stop[i].tv_nsec - watch[i].tv_nsec)/1e9;
    printf("[elapsed time] %f\n", acc/BENCHSIZE);
    #endif
    
    cliVertexTransform.Release();
    cliBitonicZSort.Release();
    cliComputeNormals.Release();

    return 0;
}


