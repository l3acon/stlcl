
// STL fie I/O
// very quick implimenation that isn't 
// all correct

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cmath>

#include "stl.hpp"


//  number of data types in a STL facet
#define FLOATS_PER_FACET 12
#define NORMALS_PER_FACET 3
#define VERTICES_PER_FACET 9

//  floating point data valuess in a transform
#define TRANSFORM_FLOATS 16

//  tolerance when comparing floatig point numbers
//  the CPU fails the transformation test at TOL > 10e-6
#define TOL 10e-5

unsigned int  stlRead(
    const char* stlFile, 
    std::vector<float> &verticies, 
    std::vector<float> &normals)
{
    float facet[TRANSFORM_FLOATS];

    FILE *ifile = fopen(stlFile, "rb");

    if (!ifile)
    {
        std::cout<<"ERROR reading file"<<std::endl;
        return 1;
    }

    char title[80];
    unsigned int nFaces;

    fread(title,80,1,ifile);
    fread( (void *) &nFaces, 4,1,ifile);
		verticies.reserve(nFaces*VERTICES_PER_FACET*sizeof(float));
		normals.reserve(nFaces*NORMALS_PER_FACET*sizeof(float));
    for (size_t i = 0; i < nFaces*FLOATS_PER_FACET; i+=FLOATS_PER_FACET)
    {
        fread( (void*) facet, sizeof(float), FLOATS_PER_FACET, ifile);
        for(size_t j = 0; j < NORMALS_PER_FACET; j++)
            normals.push_back(facet[j]);

        for(size_t j = NORMALS_PER_FACET; j < FLOATS_PER_FACET; ++j)
            verticies.push_back(facet[j]);
        fread(facet, sizeof(unsigned short), 1, ifile);
    }
    return nFaces;
}

int compareToFile(
    const char* ifileName, 
    float *buffer,
    int n)
{
    std::ifstream ifile;
    ifile.open(ifileName, std::fstream::in);

    float tmp = 0;
    int rtn = 1;

    if(ifile.is_open())
    {
        for (int i = 0; ifile >> tmp; ++i)
        {
            if(fabs(tmp - buffer[i]) > TOL || i > n)
            {
                printf("%d : c%f o%f\n", i, buffer[i], tmp);
                rtn = 0 ;
            }
        }
        return rtn;
    }
    return 0;

}

int stlWrite(
    const char* stlFile, 
    std::vector<float> &verticies, 
    std::vector<float> &normals);

//
//	transforms
//

void VertexTransform(
    float *xMat,
    float *vi,
    float *verto,
	float n)
{
	for(int i = 0; i < n; i+=VERTICES_PER_FACET)
	{
        // do the matTransform

        // x1-3 = 0 3 6
        // y1-3 = 1 4 7
        // z1-3 = 2 5 8

        //Vertex Transform 2.0
        //Now with a 4x4 matrix!

        verto[i+0] = xMat[0]*vi[i+0] + xMat[4]*vi[i+1] + xMat[8]* vi[i+2] + xMat[12];
        verto[i+1] = xMat[1]*vi[i+0] + xMat[5]*vi[i+1] + xMat[9]* vi[i+2] + xMat[13];
        verto[i+2] = xMat[2]*vi[i+0] + xMat[6]*vi[i+1] + xMat[10]*vi[i+2] + xMat[14];
    
        verto[i+3] = xMat[0]*vi[i+3] + xMat[4]*vi[i+4] + xMat[8]*vi [i+5] + xMat[12];
        verto[i+4] = xMat[1]*vi[i+3] + xMat[5]*vi[i+4] + xMat[9]*vi [i+5] + xMat[13];
        verto[i+5] = xMat[2]*vi[i+3] + xMat[6]*vi[i+4] + xMat[10]*vi[i+5] + xMat[14];
            
        verto[i+6] = xMat[0]*vi[i+6] + xMat[4]*vi[i+7] + xMat[8]*vi [i+8] + xMat[12];
        verto[i+7] = xMat[1]*vi[i+6] + xMat[5]*vi[i+7] + xMat[9]*vi [i+8] + xMat[13];
        verto[i+8] = xMat[2]*vi[i+6] + xMat[6]*vi[i+7] + xMat[10]*vi[i+8] + xMat[14];

	}
}


void ComputeNormals(                             
    float *vi,                    
    float *no,
	int n)
{
	int ii = 0;         

	for(int io = 0; io < n; io+=NORMALS_PER_FACET)
	{
		ii =io*3;
	
		//fairly sure this actually works              
		float t[4];                                    
		t[0] = (vi[ii+4]-vi[ii+3])*(vi[ii+8]-vi[ii+6]) - (vi[ii+7]-vi[ii+6])*(vi[ii+5]-vi[ii+3]); 			
		t[1] = (vi[ii+7]-vi[ii+6])*(vi[ii+2]-vi[ii+0]) - (vi[ii+1]-vi[ii+0])*(vi[ii+8]-vi[ii+6]); 			
		t[2] = (vi[ii+1]-vi[ii+0])*(vi[ii+5]-vi[ii+3]) - (vi[ii+4]-vi[ii+3])*(vi[ii+2]-vi[ii+0]); 			
		t[3] = t[0]+t[1]+t[2];                         
	
		//	probably do better error checking
		no[io  ] = t[3] ? t[0]/t[3] : 0;                           
		no[io+1] = t[3] ? t[1]/t[3] : 0;                          
		no[io+2] = t[3] ? t[2]/t[3] : 0;                          
	}
}                                                  


