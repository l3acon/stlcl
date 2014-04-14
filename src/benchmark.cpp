
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


#define CL_STATS 1

#ifndef _WIN32
#ifndef __APPLE__
#define TIME 1
#define BENCHSIZE 16
#endif
#endif

int main() 
{
    const char* stlFile = "MiddleRioGrande_Final_OneInchSpacing.stl";

    std::vector<float> verticies;
    std::vector<float> normals;
    
    float A[12];
		unsigned int facets;

    //initalize our transform matrix naively
    for (int i = 0; i < 12; ++i)
        A[i] = (float) i;

    //file stuff
    if(-1 ==  (facets = stlRead(stlFile, verticies, normals)))
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
		printf("Triangles: %d\n", (int) facets);

    // set up CLInterface (CL context/devices)
    CLI stlcl;

    // build opencl kernels and programs
    // this is kind of messy
    OCLS vertexTransform(
        stl_cl_vertexTransform_kernel_source,
        "_kVertexTransform",
        stlcl.context,
        stlcl.devices,
        stlcl.numDevices);
    stlcl.kernels.push_back(vertexTransform.kernel);
    const int vtKernel_Descriptor = 0;


    OCLS bitonicZSort(
        bitonic_STL_sort_source,
        "_kBitonic_STL_Sort",
        stlcl.context,
        stlcl.devices,
        stlcl.numDevices);
    stlcl.kernels.push_back(bitonicZSort.kernel);
    const int bzsKernel_Descriptor = 1;


    OCLS computeNormals(
        stl_cl_computeNormals_kernel_source,
        "_kComputeNormals",
        stlcl.context,
        stlcl.devices,
        stlcl.numDevices);
    stlcl.kernels.push_back(computeNormals.kernel);
    const int cnKernel_Descriptor = 2;


    #if CL_STATS
    printf("VTz:\n");
    vertexTransform.PrintErrors();
    printf("BZSz:\n");
    bitonicZSort.PrintErrors();
    printf("CNz:\n");
    computeNormals.PrintErrors();
    #endif

    // do the benchmark
    #if TIME
    timespec watch[BENCHSIZE], stop[BENCHSIZE];
    for (int i = 0; i < BENCHSIZE; ++i)
    {
        clock_gettime(CLOCK_REALTIME, &watch[i]);
    #endif

        //printf("vert: %d norm: %d \n",verticies.size(), normals.size() );

        // allocate buffers for our data output
        float* vertexBuffer = (float*) malloc(sizeof(float) * verticies.size());
        float* normalBuffer = (float*) malloc(sizeof(float) * verticies.size()/3);
        
        // padd the verticies for our sort
        stlcl.TwosPad(verticies);

        // do the transform
        stlcl.VertexTransform(
            &A[0], 
            verticies,
            vtKernel_Descriptor); 
        stlcl.Finish();        //block till done
        printf("VT done\n");

        //  sort on Z's
        stlcl.Sort(bzsKernel_Descriptor);
        stlcl.Finish();        //block till done
        printf("Sort done\n");
        
        //  buffer back the vertices
        stlcl.EnqueueUnpaddedVertexBuffer(vertexBuffer);

        //  compute normal vectors
        int cnDes = stlcl.ComputeNormals(
            verticies.size(), 
            //normalBuffer, 
            CL_TRUE,                //blocking
            cnKernel_Descriptor);

        stlcl.Finish();        //block till done
        printf("CN done\n");
        
        // not entirely working yet
        //stlcl.EnqueueUnpaddedNormalBuffer(cnDes, normalBuffer);
        //stlcl.Finish();

        //for (size_t k = 2; k < cli.original_vertex_size; k+=9)
        //{
        //    printf("i=%d: %f\n", k, vertexBuffer[k]);
        //}

        #if CL_STATS
        printf("all:\n");
        stlcl.PrintErrors();
        //printf("_kbitonic_STL_Sort:\n");
        //cli.bitonicZSort.PrintErrors();
        //printf("_kComputeNormal:\n");
        //cli.computeNormals.PrintErrors();
        #endif
        
        free(vertexBuffer);
        free(normalBuffer);
        stlcl.ReleaseDeviceMemory();
    #if TIME    
        clock_gettime(CLOCK_REALTIME, &stop[i]); // Works on Linux but not OSX
    }

    double acc = 0.0;
    for (int i = 0; i < BENCHSIZE; ++i)
        acc += stop[i].tv_sec - watch[i].tv_sec + (stop[i].tv_nsec - watch[i].tv_nsec)/1e9;
    printf("[elapsed time] %f\n", acc/BENCHSIZE);
    #endif
    
    stlcl.Release();
    vertexTransform.Release();

    return 0;
}


