#ifndef CLI_H
#define CLI_H


// opencl interface
// the wrappers make writing code in 
// OpenCL cleaner and make it easier
// to keep track of OpenCL data

#include <cstdio>
#include <cstdlib>
#include <vector>

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

#define STATUS_CHAR_SIZE 35

using namespace std;

class CLI
{
public:

    const char* programSource;
    const char * kernel_name;

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
    // used for 7: Create and compile the program     
    cl_program program;
    // used for 8: Create the kernel
    cl_kernel kernel;
    // internal status to check the output of each API call
    std::vector<cl_int> errors;
    //I think the vectors need to be constructed, even in a struct
    //std::vector<cl_mem> clMemDes;

    CLI ( const char* kernalSource,  const char* kernalName) 
    {
        programSource = kernalSource; 
        kernel_name = kernalName;
    }

// Initialize our CLI wrapper
// get OpenCL platform IDs
// get OpenCL device IDs
// create OpenCL Context and Command Queue
void Initialize()
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


// wrapper for building OpenCL program
// 
void Build ()
{
    cl_int localstatus;

    //-----------------------------------------------------
    // STEP 7: Create and compile the program
    //----------------------------------------------------- 

    // Create a program using clCreateProgramWithSource()
    program = clCreateProgramWithSource(
        context, 
        1, 
        (const char**)&programSource,                                 
        NULL, 
        &localstatus);
    errors.push_back(localstatus);

    // Build (compile) the program for the devices with
    // clBuildProgram()
    localstatus = clBuildProgram(
        program, 
        numDevices, 
        devices, 
        NULL, 
        NULL, 
        NULL);
    
    errors.push_back(localstatus);

    //-----------------------------------------------------
    // STEP 8: Create the kernel
    //----------------------------------------------------- 
    // Use clCreateKernel() to create a kernel from the 
    // vector addition function (named "vecadd")
    kernel = clCreateKernel(program, kernel_name , &localstatus);
    errors.push_back(localstatus);

    return;
}

// wraper function for kernel arguments
// this reduces the code required to
// setup buffers and set arguments
cl_mem KernelArgs(
    void* ptr,              // I want to restrict this
    size_t bufferBytes,  
    int argn, 
    cl_mem_flags memflag)
{
    cl_int localstatus;
    cl_mem clmemDes;

    // could have issues with this for .hpp version
    clmemDes = clCreateBuffer(
        context,
        memflag,
        bufferBytes,
        NULL,
        &localstatus);

    if(memflag == CL_MEM_READ_ONLY)
        localstatus = clEnqueueWriteBuffer(
            cmdQueue,
            clmemDes,
            CL_FALSE,
            0,
            bufferBytes,
            ptr,
            0,
            NULL,
            NULL);
    localstatus = clSetKernelArg(
        kernel,
        argn,
        sizeof(cl_mem),
        &clmemDes);

    errors.push_back(localstatus);

    return clmemDes;
}

// release CLI memory
void Release()
{
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);
    free(platforms);
    free(devices);

}

// translate OpenCL status codes to human
// readable errors
void Errors(const cl_int err, char* stat)
{
    //printing is inefficient anyway
    const char zero[STATUS_CHAR_SIZE] = {};
    strcpy(stat,zero);
    switch (err) 
    {
        case CL_SUCCESS:                            strcpy(stat, "Success!"); break;
        case CL_DEVICE_NOT_FOUND:                   strcpy(stat, "Device not found."); break;
        case CL_DEVICE_NOT_AVAILABLE:               strcpy(stat, "Device not available"); break;
        case CL_COMPILER_NOT_AVAILABLE:             strcpy(stat, "Compiler not available"); break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      strcpy(stat, "Memory object allocation failure"); break;
        case CL_OUT_OF_RESOURCES:                   strcpy(stat, "Out of resources"); break;
        case CL_OUT_OF_HOST_MEMORY:                 strcpy(stat, "Out of host memory"); break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:       strcpy(stat, "Profiling information not available"); break;
        case CL_MEM_COPY_OVERLAP:                   strcpy(stat, "Memory copy overlap"); break;
        case CL_IMAGE_FORMAT_MISMATCH:              strcpy(stat, "Image format mismatch"); break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         strcpy(stat, "Image format not supported"); break;
        case CL_BUILD_PROGRAM_FAILURE:              strcpy(stat, "Program build failure"); break;
        case CL_MAP_FAILURE:                        strcpy(stat, "Map failure"); break;
        case CL_INVALID_VALUE:                      strcpy(stat, "Invalid value"); break;
        case CL_INVALID_DEVICE_TYPE:                strcpy(stat, "Invalid device type"); break;
        case CL_INVALID_PLATFORM:                   strcpy(stat, "Invalid platform"); break;
        case CL_INVALID_DEVICE:                     strcpy(stat, "Invalid device"); break;
        case CL_INVALID_CONTEXT:                    strcpy(stat, "Invalid context"); break;
        case CL_INVALID_QUEUE_PROPERTIES:           strcpy(stat, "Invalid queue properties"); break;
        case CL_INVALID_COMMAND_QUEUE:              strcpy(stat, "Invalid command queue"); break;
        case CL_INVALID_HOST_PTR:                   strcpy(stat, "Invalid host pointer"); break;
        case CL_INVALID_MEM_OBJECT:                 strcpy(stat, "Invalid memory object"); break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    strcpy(stat, "Invalid image format descriptor"); break;
        case CL_INVALID_IMAGE_SIZE:                 strcpy(stat, "Invalid image size"); break;
        case CL_INVALID_SAMPLER:                    strcpy(stat, "Invalid sampler"); break;
        case CL_INVALID_BINARY:                     strcpy(stat, "Invalid binary"); break;
        case CL_INVALID_BUILD_OPTIONS:              strcpy(stat, "Invalid build options"); break;
        case CL_INVALID_PROGRAM:                    strcpy(stat, "Invalid program"); break;
        case CL_INVALID_PROGRAM_EXECUTABLE:         strcpy(stat, "Invalid program executable"); break;
        case CL_INVALID_KERNEL_NAME:                strcpy(stat, "Invalid kernel name"); break;
        case CL_INVALID_KERNEL_DEFINITION:          strcpy(stat, "Invalid kernel definition"); break;
        case CL_INVALID_KERNEL:                     strcpy(stat, "Invalid kernel"); break;
        case CL_INVALID_ARG_INDEX:                  strcpy(stat, "Invalid argument index"); break;
        case CL_INVALID_ARG_VALUE:                  strcpy(stat, "Invalid argument value"); break;
        case CL_INVALID_ARG_SIZE:                   strcpy(stat, "Invalid argument size"); break;
        case CL_INVALID_KERNEL_ARGS:                strcpy(stat, "Invalid kernel arguments"); break;
        case CL_INVALID_WORK_DIMENSION:             strcpy(stat, "Invalid work dimension"); break;
        case CL_INVALID_WORK_GROUP_SIZE:            strcpy(stat, "Invalid work group size"); break;
        case CL_INVALID_WORK_ITEM_SIZE:             strcpy(stat, "Invalid work item size"); break;
        case CL_INVALID_GLOBAL_OFFSET:              strcpy(stat, "Invalid global offset"); break;
        case CL_INVALID_EVENT_WAIT_LIST:            strcpy(stat, "Invalid event wait list"); break;
        case CL_INVALID_EVENT:                      strcpy(stat, "Invalid event"); break;
        case CL_INVALID_OPERATION:                  strcpy(stat, "Invalid operation"); break;
        case CL_INVALID_GL_OBJECT:                  strcpy(stat, "Invalid OpenGL object"); break;
        case CL_INVALID_BUFFER_SIZE:                strcpy(stat, "Invalid buffer size"); break;
        case CL_INVALID_MIP_LEVEL:                  strcpy(stat, "Invalid mip-map level"); break;
    }
    return; 
}

// print all CLI status/errors
//
void PrintErrors()
{
    char tmp[STATUS_CHAR_SIZE];

    for( std::vector<cl_int>::const_iterator i = errors.begin(); i != errors.end(); ++i)
    {
        Errors(*i, tmp);
        printf("%s\n", tmp);
    }
}


};




#endif