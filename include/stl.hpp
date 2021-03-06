#ifndef STL_H
#define STL_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>


#define STL_TRANSFORM_MATRIX_SIZE 12


typedef struct xformmat_s
{	
	static const int size = STL_TRANSFORM_MATRIX_SIZE;
	float stlTransformMatrix[STL_TRANSFORM_MATRIX_SIZE];
} XformMat;

unsigned int stlRead(const char* stlFile, 
    std::vector<float> &verticies, 
    std::vector<float> &normals);

int stlWrite(const char* stlFile, 
    std::vector<float> &verticies, 
    std::vector<float> &normals);

int compareToFile(
    const char* ifileName, 
    float *buffer,
    int n);

float stlVerifyTransform(const float* xMat, 
    float* v, float* xformedv, 
    unsigned int nFaces);

void ComputeNormals(                             
            float *vi,                    
            float *no,
						int n);

void VertexTransform(            
            float *xMat,    
            float *vi,      
            float *verto, 
						float n);


#endif
