
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
#include "ocls.hpp"


#define CL_ERRORS 0

#ifndef _WIN32
#ifndef __APPLE__
#define TIME 1
#define BENCHSIZE 16 
#endif
#endif

int main() 
{
    const char* stlFile = "Engine_Block.stl";

    std::vector<float> verticies;
    std::vector<float> normals;
    
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
    CLI cli;

    OCLS vertexTransform(stl_cl_vertexTransform_kernel_source,
        "_kVertexTransform",
        cli.context,
        cli.devices,
        cli.numDevices);
    cli.kernels.push_back(vertexTransform.kernel);
    printf("VTz:\n");


    OCLS bitonicZSort(
        bitonic_STL_sort_source,
        "_kBitonic_STL_Sort",
        cli.context,
        cli.devices,
        cli.numDevices);
    cli.kernels.push_back(bitonicZSort.kernel);
    printf("BZSz:\n");

    OCLS computeNormals(
        stl_cl_computeNormals_kernel_source,
        "_kComputeNormals",
        cli.context,
        cli.devices,
        cli.numDevices);
    cli.kernels.push_back(computeNormals.kernel);
    printf("CNz:\n");

    #if CL_ERRORS
    vertexTransform.PrintErrors();
    bitonicZSort.PrintErrors();
    computeNormals.PrintErrors();
    #endif
    // do the benchmark
    #if TIME
    timespec watch[BENCHSIZE], stop[BENCHSIZE];
    for (int i = 0; i < BENCHSIZE; ++i)
    {
        clock_gettime(CLOCK_REALTIME, &watch[i]);
    #endif        
        
        cli.TwosPad(verticies);

        float* vertexBuffer = (float*) malloc(sizeof(float) * verticies.size());
        float* normalBuffer = (float*) malloc(sizeof(float) * verticies.size()/3);

        cli.VertexTransform(
            &A[0], 
            verticies,
            0);

        cli.Finish();
        printf("VT done\n");

        cli.Sort(1);
        cli.Finish();
        printf("Sort done\n");

        cli.ComputeNormals(
            verticies.size(), 
            normalBuffer, 
            CL_TRUE,
            2);
        cli.Finish();
        printf("CN done\n");
        
        cli.EnqueueUnpaddedBuffer(vertexBuffer);


        //for (size_t k = 2; k < cli.original_vertex_size; k+=9)
        //{
        //    printf("i=%d: %f\n", k, vertexBuffer[k]);
        //}

        #if CL_ERRORS
        printf("all:\n");
        cli.PrintErrors();
        //printf("_kbitonic_STL_Sort:\n");
        //cli.bitonicZSort.PrintErrors();
        //printf("_kComputeNormal:\n");
        //cli.computeNormals.PrintErrors();
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
    
    cli.Release();
    vertexTransform.Release();

    return 0;
}


