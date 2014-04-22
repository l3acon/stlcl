
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

// opencl interface
// the wrappers make writing code in 
// OpenCL cleaner and make it easier
// to keep track of OpenCL data

#include "cli.hpp"

#define WORK_GROUP_SIZE 128 // logical errors occur after work group size > 128
#define VERTEX_FLOATS 9
#define TRANSFORM_SIZE 16

#define STATUS_CHAR_SIZE 35

using namespace std;

void CLI::Finish()
    {    clFinish(cmdQueue);    }



// wraper function for kernel arguments
// this reduces the code required to
// setup buffers and set arguments
cl_mem CLI::KernelArgs(
    void* ptr,              // I want to restrict this
    size_t bufferBytes,  
    int argn, 
    cl_mem_flags memflag,
    unsigned int kernelIndex)
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

    if(memflag == CL_MEM_READ_ONLY || memflag == CL_MEM_READ_WRITE)
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
        kernels[kernelIndex],
        argn,
        sizeof(cl_mem),
        &clmemDes);

    errors.push_back(localstatus);
    return clmemDes;
}

void CLI::ReleaseDeviceMemory()
{
    //release all cl_mem memory discriptor objects
    for( std::vector<cl_mem>::const_iterator i = cl_memory_descriptors.begin(); 
        i != cl_memory_descriptors.end(); ++i)
    {
        clReleaseMemObject(*i);
    }

    cl_memory_descriptors.clear();

}

// release CLI memory
void CLI::Release()
{
    ReleaseDeviceMemory();
    //clReleaseKernel(kernels[i]);
    //clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);
    free(platforms);
    free(devices);
}

// translate OpenCL status codes to human
// readable errors
void CLI::Errors(const cl_int err, char* stat)
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

void CLI::PrintErrors()
{
    char tmp[STATUS_CHAR_SIZE];

    for( std::vector<cl_int>::const_iterator i = errors.begin(); i != errors.end(); ++i)
    {
        if(*i)
        {
            Errors(*i, tmp);
            printf("%s\n", tmp);

        }
    }
    errors.clear();
}


void CLI::PrintStats()
{
    char tmp[STATUS_CHAR_SIZE];

    for( std::vector<cl_int>::const_iterator i = errors.begin(); i != errors.end(); ++i)
    {
        Errors(*i, tmp);
        printf("%s\n", tmp);
    }
    errors.clear();
}


int CLI::ComputeNormals(
    unsigned int nVerticies,
    int cli_flags,
    unsigned int kernelIndex)
{
    cl_int localstatus;

    //size_t vertexBytes = sizeof(float)*12;
    //size_t vertexBytes = nVerticies * sizeof(float);
    size_t normalBytes = (nVerticies * sizeof(float))/3;

    //initalize CL interface and build kernel
    // declare CL memory buffers

    clSetKernelArg(
        kernels[kernelIndex], 
        0, 
        sizeof(cl_mem), 
        (void*) &cl_memory_descriptors[0]);

    cl_memory_descriptors.push_back( 
        clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            normalBytes,
            NULL,
            &localstatus)
    );
    errors.push_back(localstatus);
     int output_memory_descriptor = cl_memory_descriptors.size()-1;

    localstatus = clSetKernelArg(
            kernels[kernelIndex],
            1,
            sizeof(cl_mem),
            (void *) &cl_memory_descriptors[output_memory_descriptor]); 

    errors.push_back(localstatus);

    output_memory_descriptor = cl_memory_descriptors.size() - 1;

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
        kernels[kernelIndex], 
        1, 
        NULL, 
        globalWorkSize, 
        NULL, 
        0, 
        NULL, 
        NULL);

    errors.push_back(localstatus);

    //clEnqueueReadBuffer(
    //    cmdQueue, 
    //    cl_memory_descriptors[1], 
    //    cli_flags,            // CL_TRUE is a BLOCKING read
    //    0, 
    //    normalBytes, 
    //    normalBuffer, 
    //    0, 
    //    NULL, 
    //    NULL);

    // Free OpenCL resources
    //clReleaseMemObject(buffer);

    // Free host resources

    return output_memory_descriptor;
}

void CLI::VertexTransform(
    float* transform, 
    std::vector<float> &verticies,
    unsigned int kernelIndex)
{
    cl_int localstatus;
    unsigned int nVerticies = verticies.size();
    size_t vertexBytes = nVerticies * sizeof(float);
    
    float transformArray[TRANSFORM_SIZE];
    for (int i = 0; i < TRANSFORM_SIZE; ++i)
        transformArray[i] = transform[i]; 

    cl_mem bufferA = KernelArgs(
        transformArray,
        TRANSFORM_SIZE*sizeof(float),
        0,
        CL_MEM_READ_ONLY,
        kernelIndex);

    cl_mem bufferB = KernelArgs(
        &verticies.front(),
        vertexBytes,
        1,
        CL_MEM_READ_WRITE,
        kernelIndex);

    cl_memory_descriptors.push_back( 
        clCreateBuffer(
            context, 
            CL_MEM_READ_WRITE, 
            vertexBytes, 
            NULL,
            &localstatus)
    );

    clSetKernelArg(
        kernels[kernelIndex], 
        2, 
        sizeof(cl_mem), 
        (void *) &cl_memory_descriptors[0]); 

    errors.push_back(localstatus);

/*
    cl_mem bufferC = KernelArgs(
        vertexBuffer,
        vertexBytes,
        2,
        CL_MEM_WRITE_ONLY);

*/
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
        kernels[kernelIndex], 
        1, 
        NULL, 
        globalWorkSize, 
        NULL, 
        0, 
        NULL, 
        NULL);

    errors.push_back(localstatus);
    
    // Free OpenCL resources
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    //clReleaseMemObject(bufferC);

    // Free host resources

    return;
}


bool CLI::IsPowerOfTwo(unsigned long x)
    {   return (x & (x - 1)) == 0;  }

void CLI::TwosPad(std::vector<float> &verticies)
{
    //  --------------------------
    //
    // pad our verticies with -1's
    //
    //  --------------------------
    unsigned int n = verticies.size()/VERTEX_FLOATS - 1;
    unsigned int p2 = 0;
    
    original_vertex_size = verticies.size();
    do ++p2; while( (n >>= 0x1) != 0);
    padded_size = 0x1 << p2;

    // it just needs to be larger really
    // I don't know if CPP can do this
    // in an efficient way
    start_of_padding = verticies.end();
    verticies.insert(
        verticies.end(), 
        padded_size*9 - verticies.size(),
        0.0); 	//defined in math.h
}

// void RemovePad(std::vector<float> &verticies)
//     {   verticies.erase(start_of_padding, verticies.end() );   }


void  CLI::EnqueueUnpaddedVertexBuffer(int n, float* vertices )
{
    //printf("vf: %d\n",(padded_size*9 - original_vertex_size ) );
   clEnqueueReadBuffer(
        cmdQueue, 
        cl_memory_descriptors[0], 
        CL_FALSE,        // CL_TRUE is a BLOCKING read
        0, 
        n*sizeof(float), 
        vertices, 
        0, 
        NULL, 
        NULL);
return ;
}

//  removes padding 
void  CLI::EnqueuePaddedVertexBuffer(float* vertices )  
{
    //printf("vf: %d\n",(padded_size*9 - original_vertex_size ) );
   clEnqueueReadBuffer(
        cmdQueue, 
        cl_memory_descriptors[0], 
        CL_FALSE,        // CL_TRUE is a BLOCKING read
        (padded_size*9 - original_vertex_size )*sizeof(float), 
        original_vertex_size*sizeof(float), 
        vertices, 
        0, 
        NULL, 
        NULL);
return ;
}


void  CLI::EnqueueUnpaddedNormalBuffer(int des, int n, float* normals )
{
        // should do some sanity checks
        // like for everything

    //printf("cf: %d\n", (padded_size*3 - original_vertex_size/3 ) );
    //printf("osize: %d\n", original_vertex_size);

    clEnqueueReadBuffer(
         cmdQueue, 
         cl_memory_descriptors[des], 
         CL_FALSE,        // CL_TRUE is a BLOCKING read
         0, 
         n*sizeof(float), 
         normals, 
         0, 
         NULL, 
         NULL);


    return ;
}

//  removes padding (might not work)
void  CLI::EnqueuePaddedNormalBuffer(int des, float* normals )
{
        // should do some sanity checks
        // like for everything

    //printf("cf: %d\n", (padded_size*3 - original_vertex_size/3 ) );
    //printf("osize: %d\n", original_vertex_size);

    clEnqueueReadBuffer(
         cmdQueue, 
         cl_memory_descriptors[des], 
         CL_FALSE,        // CL_TRUE is a BLOCKING read
         (padded_size*3 - original_vertex_size/3 ) * sizeof(float), 
         original_vertex_size*sizeof(float)/3, 
         normals, 
         0, 
         NULL, 
         NULL);


    return ;
}



void CLI::Sort(unsigned int kernelIndex)
{
    cl_int local_status;
    size_t global_size = padded_size/2;
    size_t local_size = WORK_GROUP_SIZE;
			//size_t local_size = global_size;
    //size_t num_of_work_groups = global_size/local_size;

    clSetKernelArg(
        kernels[kernelIndex], 
        0, 
        sizeof(cl_mem), 
        (void *) &cl_memory_descriptors[0]);
    
    unsigned int stage, passOfStage, numStages, temp;
    stage = passOfStage = numStages = 0;
    
    for(temp = padded_size; temp > 1; temp >>= 1)
        ++numStages;
 
    global_size = padded_size>>1;
    local_size = WORK_GROUP_SIZE;
    
    for(stage = 0; stage < numStages; ++stage)
    {
        // stage of the algorithm
        clSetKernelArg(
            kernels[kernelIndex], 
            1, 
            sizeof(int), 
            (void *)&stage);

        // Every stage has stage + 1 passes
        for(passOfStage = 0; passOfStage < stage + 1; 
            ++passOfStage) 
        {
            // pass of the current stage
            #if ERRORS
            printf("Pass no: %d\n",passOfStage);
            #endif
            local_status = clSetKernelArg(
                kernels[kernelIndex], 
                2,  
                sizeof(int), 
                (void *)&passOfStage);
                
                errors.push_back(local_status);
            //
            // Enqueue a kernel run call.
            // Each thread writes a sorted pair.
            // So, the number of threads (global) should be half the 
            // length of the input buffer.
            //
            clEnqueueNDRangeKernel(
                cmdQueue, 
                kernels[kernelIndex], 
                1, 
                NULL,
                &global_size, 
                &local_size, 
                0, 
                NULL, 
                NULL);  

            clFinish(cmdQueue);
        } //end of for passStage = 0:stage-1
    } //end of for stage = 0:numStage-1

}


