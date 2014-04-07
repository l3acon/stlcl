#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "cli.hpp"
#include "kernels.hpp"
#include "stl.hpp"

using namespace std;

#define CL_ERRORS 1
//#define DATA_TYPE int 
//#define DATA_SIZE 1024
#define WORK_GROUP_SIZE 64 // logical errors occur after work group size > 128


class stlclBitonicZSort : public CLI
{
public:
    unsigned int padd;
    size_t original_vertex_size;
    size_t padded_size;

    stlclBitonicZSort(
            const char* kernalSource, 
            const char* kernalName) : CLI ( kernalSource, kernalName)
    {}

bool IsPowerOfTwo(unsigned long x)
{
    return (x & (x - 1)) == 0;
}

void TwosPadd(std::vector<float> verticies)
{
    //  --------------------------
    //
    // pad our verticies with -1's
    //
    //  --------------------------
    unsigned int n = verticies.size()/9 - 1;
    unsigned int p2 = 0;
    
    original_vertex_size = verticies.size();
    do ++p2; while( (n >>= 0x1) != 0);
    padded_size = 0x1 << p2;

    padd = 0;
    // it just needs to be larger really
    // I don't know if CPP can do this
    // in an efficient way
    while(verticies.size()/9 < padded_size)
    {
        verticies.push_back(-1.0);
        ++padd;
    }
}


int Sort(
    std::vector<float> &verticies)
{
    cl_int local_status;
    if(!IsPowerOfTwo( verticies.size() ))
        return -1;
    
    padded_size = verticies.size();

    //  --------------------------
    //
    // OpenCL stuff
    //
    //  --------------------------
    
    // Basic initialization and declaration...
    // Execute the OpenCL kernel on the list
    // Each work item shall compare two elements.
    size_t global_size = padded_size/2;
    // This is the size of the work group.
    size_t local_size = WORK_GROUP_SIZE;
     // Calculate the Number of work groups.
    size_t num_of_work_groups = global_size/local_size;

    //Create memory buffers on the device for each vector
    cl_mem pInputBuffer_clmem = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 
        padded_size * sizeof(Vertex), 
        (Vertex*) &verticies.front(),
        &local_status);

  	errors.push_back(local_status); 
    // create kernel
    
    clSetKernelArg(
        kernel, 
        0, 
        sizeof(cl_mem), 
        (void *) &pInputBuffer_clmem);

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
            kernel, 
            1, 
            sizeof(int), 
            (void *)&stage);

        // Every stage has stage + 1 passes
        for(passOfStage = 0; passOfStage < stage + 1; 
            ++passOfStage) 
        {
            // pass of the current stage
            printf("Pass no: %d\n",passOfStage);
            local_status = clSetKernelArg(
                kernel, 
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
                kernel, 
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
 
    Vertex *mapped_input_buffer =
        (Vertex *)clEnqueueMapBuffer(
            cmdQueue, 
            pInputBuffer_clmem, 
            true, 
            CL_MAP_READ, 
            0, 
            sizeof(Vertex) * padded_size, 
            0, 
            NULL, 
            NULL, 
            &local_status);

		errors.push_back(local_status);


    //  --------------------------
    //
    // Done
    //
    //  --------------------------

    std::vector<float> output;

    for (int i = padd; i < padded_size; ++i)
    {
        /* code */
    }

    // this is inefficient
    int count=0; 
    for (int i = 0; i < verticies.size(); ++i)
    {
        if(verticies[i] == -1.0)
            count++;
    }

    //Display the Sorted data on the screenm
    for(int i = 0; i < padded_size; i++)
    {
        printf("i: %d : %f\n",i, mapped_input_buffer[i].x1 );   
        printf("i: %d : %f\n",i, mapped_input_buffer[i].y1 );    
        printf("i: %d : %f\n",i, mapped_input_buffer[i].z1 );    
        printf("i: %d : %f\n",i, mapped_input_buffer[i].x2 );   
        printf("i: %d : %f\n",i, mapped_input_buffer[i].y2 );    
        printf("i: %d : %f\n",i, mapped_input_buffer[i].z2 );    
        printf("i: %d : %f\n",i, mapped_input_buffer[i].x3 );   
        printf("i: %d : %f\n",i, mapped_input_buffer[i].y3 );    
        printf("i: %d : %f\n",i, mapped_input_buffer[i].z3 );    

    }
    printf("think2: %d\n", padded_size*9 - padd);
		
    // cleanup...
    return 0;
}

};
