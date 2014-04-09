#ifndef OCLS_H
#define OCLS_H


#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string.h>
#include "kernels.hpp"

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


class OCLS
{
public:

	// used for 7: Create and compile the program     
    cl_program program;
    // used for 8: Create the kernel
    cl_kernel kernel;
    // internal status to check the output of each API call
    std::vector<cl_int> errors;
    //I think the vectors need to be constructed, even in a struct
    //std::vector<cl_mem> clMemDes;

    // wrapper for building OpenCL program
    // 
    OCLS (
    	const char* programSource,  
    	const char* kernel_name, 
    	cl_context context,
    	cl_device_id *devices,
    	cl_uint numDevices)

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

    void Release()
    {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
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
