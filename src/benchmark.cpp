
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
#include "qsorting.hpp"

#include "cli.hpp"
#include "ocls.hpp"

#define SORTED_TRANSFORM 0
#define CL_STATS 1

#ifndef _WIN32
#ifndef __APPLE__
#define TIME 1
#define BENCHSIZE 1
#endif
#endif

#define XFORM_FLOATS 16

int main() 
{
    const char* stlFile = "Pyramid.stl";

    std::vector<float> verticies;
    std::vector<float> normals;
    
		int facets;

   	// 30 degree rotation along X axis 
    // also translation of (0.1, 0.2, 0.5) in X, Y and Z direction.
    // the matrix is row major 
    float A[XFORM_FLOATS] = {
        1.0,
        0.0,
        0.0,
        0.0,

        0.0   ,
        0.86602540378 ,
        0.5   ,
        0.0   ,

        0.0  ,
        -0.5   ,
        0.86602540378,
        0.0  ,
        
        0.1,
        0.2,
        0.5,
        1.0
        };


    //file stuff
    if(-1 ==  (facets = stlRead(stlFile, verticies, normals)))
    {
        std::cout<<"ERROR: reading file"<<std::endl;
        return 1;
    }
		
		//printf("nfaces: %d\n vert: %d\n norm: %d\n", facets, verticies.size(), normals.size());

    //check sanity for verticies and normals
    if( fmod(verticies.size(),9.0) !=  0 || fmod(normals.size(),3.0) != 0 )
    {
        std::cout<<"ERROR: verticies and normals don't make sense"<<std::endl;
        return 1;
    }
		printf("Triangles: %d\n", (int) facets);

    if( compareToFile("orginal_vertices.txt", &verticies.front(), verticies.size()) )
        cout<< "orginal vertices test PASSED" << endl;


		// allocate buffers for our data output
    float* vertexBuffer = (float*) malloc(sizeof(float) * verticies.size());
    float* normalBuffer = (float*) malloc(sizeof(float) * verticies.size()/3);

		// do the benchmark
#if TIME
		std::cout<<"[begin] CPU BENCHMARK"<<std::endl;
    timespec watch[BENCHSIZE], stop[BENCHSIZE];
    for (int i = 0; i < BENCHSIZE; ++i)
    {
        clock_gettime(CLOCK_REALTIME, &watch[i]);
#endif

        //printf("vert: %d norm: %d \n",verticies.size(), normals.size() );
				VertexTransform(&A[0], &verticies.front(), vertexBuffer, verticies.size());
			  //qsort(vertexBuffer, verticies.size()/9, sizeof(float)*9, vertex_comparator);

				ComputeNormals(vertexBuffer, normalBuffer, normals.size());

				//for (size_t k = 4*verticies.size()/5; k < verticies.size(); k+=1)
        //{
        //    printf("i=%d: %f\n", k, vertexBuffer[k]);
        //}
				//printf("%d\n", verticies.size());
		
#if TIME    
        clock_gettime(CLOCK_REALTIME, &stop[i]); // Works on Linux but not OSX
    }

    double acc = 0.0;
    for (int i = 0; i < BENCHSIZE; ++i)
        acc += stop[i].tv_sec - watch[i].tv_sec + (stop[i].tv_nsec - watch[i].tv_nsec)/1e9;
    BENCHSIZE ? printf("[elapsed time] %f\n", acc/BENCHSIZE) : printf("Invalid BENCHSIZE\n");
#endif

		std::cout<<"CPU DONE" << std::endl;
  	if( compareToFile("transformed_vertices.txt", vertexBuffer, verticies.size()) )
    	std::cout<< "transformed vertices test PASSED" << std::endl;

		// ----------------------------------------
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
	

		// do the GPU benchmark
#if TIME
		std::cout<<"[begin] GPU  BENCHMARK"<<std::endl;
		for (int i = 0; i < BENCHSIZE; ++i)
    {
        clock_gettime(CLOCK_REALTIME, &watch[i]);
#endif

#if! SORTED_TRANSFORM

        //printf("vert: %d norm: %d \n",verticies.size(), normals.size() );
       
        // padd the verticies for our sort
        //int original_vertex_size = stlcl.TwosPad(verticies);

        // do the transform
        stlcl.VertexTransform(
            &A[0], 
            verticies,
            vtKernel_Descriptor); 
        stlcl.Finish();        //block till done
        
				//  sort on Z's
        //stlcl.Sort(bzsKernel_Descriptor);
        //stlcl.Finish();        //block till done
        
        //  buffer back the vertices
        stlcl.EnqueueUnpaddedVertexBuffer(verticies.size(), vertexBuffer);
        stlcl.Finish();

        //  compute normal vectors
        //int cnDes = stlcl.ComputeNormals(
        //    verticies.size(), 
        //    CL_TRUE,                //blocking
        //    cnKernel_Descriptor);

        //stlcl.Finish();        //block till done
        // not entirely working yet
        //stlcl.EnqueueUnpaddedNormalBuffer( verticies.size(), cnDes, normalBuffer);
				//stlcl.Finis();

#if CL_STATS
        printf("all:\n");
        stlcl.PrintStats();
        //printf("_kbitonic_STL_Sort:\n");
        //cli.bitonicZSort.PrintErrors();
        //printf("_kComputeNormal:\n");
        //cli.computeNormals.PrintErrors();
#endif

				//free(vertexBuffer);
        //free(normalBuffer);
        //stlcl.ReleaseDeviceMemory();
#endif	
#if SORTED_TRANSFORM

        if( compareToFile("transformed_vertices.txt", vertexBuffer, verticies.size()) )
        	std::cout<< "transformed vertices test PASSED" << std::endl;

        //printf("vert: %d norm: %d \n",verticies.size(), normals.size() );
       
        // padd the verticies for our sort
        int original_vertex_size = stlcl.TwosPad(verticies);

        // do the transform
        stlcl.VertexTransform(
            &A[0], 
            verticies,
            vtKernel_Descriptor); 
        stlcl.Finish();        //block till done
        
				//  sort on Z's
        stlcl.Sort(bzsKernel_Descriptor);
        stlcl.Finish();        //block till done
        
        //  buffer back the vertices
        stlcl.EnqueuePaddedVertexBuffer(vertexBuffer);
        stlcl.Finish();

        //  compute normal vectors
        //int cnDes = stlcl.ComputeNormals(
        //    verticies.size(), 
        //    CL_TRUE,                //blocking
        //    cnKernel_Descriptor);

        //stlcl.Finish();        //block till done
        // not entirely working yet
        //stlcl.EnqueueUnpaddedNormalBuffer( verticies.size(), cnDes, normalBuffer);
				//stlcl.Finis();

#if CL_STATS
        printf("all:\n");
        stlcl.PrintStats();
        //printf("_kbitonic_STL_Sort:\n");
        //cli.bitonicZSort.PrintErrors();
        //printf("_kComputeNormal:\n");
        //cli.computeNormals.PrintErrors();
#endif

				//free(vertexBuffer);
        //free(normalBuffer);
        //stlcl.ReleaseDeviceMemory();
				
        if( compareToFile("transformed_vertices.txt", vertexBuffer, verticies.size()) )
        	std::cout<< "transformed vertices test PASSED" << std::endl;
#endif


#if TIME    
        clock_gettime(CLOCK_REALTIME, &stop[i]); // Works on Linux but not OSX
    }

    acc = 0.0;
    for (int i = 0; i < BENCHSIZE; ++i)
        acc += stop[i].tv_sec - watch[i].tv_sec + (stop[i].tv_nsec - watch[i].tv_nsec)/1e9;
    BENCHSIZE ? printf("[elapsed time] %f\n", acc/BENCHSIZE) : printf("Invalid BENCHSIZE\n");
#endif

    return 0;
}


