#ifndef CLI_H
#define CLI_H


// opencl interface
// the wrappers make writing code in 
// OpenCL cleaner and make it easier
// to keep track of OpenCL data

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string.h>
#include <math.h>

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#elif __linux
#include <CL/cl.h>
#elif _WIN32
#include <CL/opencl.h>
#else
#error Platform not supported
#endif

using namespace std;

class CLI
{
public:
    std::vector<float>::iterator start_of_padding;

    size_t padded_size, original_vertex_size;

    //initialize our data we're keeping track of
    // used for  1: Discover and initialize the platforms
    cl_uint numPlatforms;
    cl_platform_id *platforms;
    // used for 2: Discover and initialize the devices
    cl_uint numDevices;

    cl_device_id *devices;
    // used for 3: Create a context
    cl_context context;
    // used for 4: Create a command queue
    cl_command_queue cmdQueue;

    std::vector<cl_kernel> kernels;

    std::vector<cl_int> errors;

    std::vector<cl_mem> cl_memory_descriptors;

    // Initialize our CLI wrapper
    // get OpenCL platform IDs
    // get OpenCL device IDs
    // create OpenCL Context and Command Queue
    CLI()
    {
        //-----------------------------------------------------
        // STEP 1: Discover and initialize the platforms
        //-----------------------------------------------------
        // Use clGetPlatformIDs() to retrieve the number of 
        // platforms
        cl_int localstatus;
        localstatus = clGetPlatformIDs(0, NULL, &numPlatforms);
        errors.push_back(localstatus);

        // Allocate enough space for each platform
        platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));

        // Fill in platforms with clGetPlatformIDs()
        localstatus = clGetPlatformIDs(numPlatforms, platforms, NULL);

        errors.push_back(localstatus);

        //-----------------------------------------------------
        // STEP 2: Discover and initialize the devices
        //----------------------------------------------------- 
        // Use clGetDeviceIDs() to retrieve the number of 
        // devices present
        localstatus = clGetDeviceIDs(
            platforms[0], 
            CL_DEVICE_TYPE_ALL, 
            0,
            NULL, 
            &numDevices);
        errors.push_back(localstatus);


        // Allocate enough space for each device
        devices = 
            (cl_device_id*)malloc(
                numDevices * sizeof(cl_device_id));

        // Fill in devices with clGetDeviceIDs()
        localstatus = clGetDeviceIDs(
            platforms[0], 
            CL_DEVICE_TYPE_ALL,
            numDevices, 
            devices, 
            NULL);
        errors.push_back(localstatus);

        //-----------------------------------------------------
        // STEP 3: Create a context
        //----------------------------------------------------- 
        
        // Create a context using clCreateContext() and 
        // associate it with the devices
        context = clCreateContext(
            NULL, 
            numDevices, 
            devices, 
            NULL, 
            NULL, 
            &localstatus);
        errors.push_back(localstatus);

        //-----------------------------------------------------
        // STEP 4: Create a command queue
        //----------------------------------------------------- 
        // Create a command queue using clCreateCommandQueue(),
        // and associate it with the device you want to execute 
        // on
        cmdQueue = clCreateCommandQueue(
            context, 
            devices[0], 
            0, 
            &localstatus);
        errors.push_back(localstatus);

        return ;
    }

    void Finish();
    void ReleaseDeviceMemory();

    // release CLI memory
    void Release();

    // translate OpenCL status codes to human
    // readable errors
    void Errors(const cl_int err, char* stat);

    void PrintErrors();

    void PrintStats();


    // wraper function for kernel arguments
    // this reduces the code required to
    // setup buffers and set arguments
    cl_mem KernelArgs(
        void* ptr,              // I want to restrict this
        size_t bufferBytes,  
        int argn, 
        cl_mem_flags memflag,
        unsigned int kernelIndex);


    int ComputeNormals(
        unsigned int nVerticies,
        int cli_flags,
        unsigned int kernelIndex);

    void VertexTransform(
        float* transform, 
        std::vector<float> &verticies,
        unsigned int kernelIndex);

    bool IsPowerOfTwo(unsigned long x);

    void TwosPad(std::vector<float> &verticies);

   // void RemovePad(std::vector<float> &verticies)
   //     {   verticies.erase(start_of_padding, verticies.end() );   }

	void  EnqueueUnpaddedVertexBuffer(float* vertices );
    void  EnqueuePaddedVertexBuffer(float* vertices );      //  removes pad

    void  EnqueueUnpaddedNormalBuffer(int des, float* normals );
    void  EnqueuePaddedNormalBuffer(int des, float* normals );  //  removes pad


    void Sort(unsigned int kernelIndex);
};




#endif
