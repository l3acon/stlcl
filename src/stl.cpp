
// STL fie I/O
// very quick implimenation that isn't 
// all correct

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

#include "stl.hpp"


#define FLOATS_PER_FACET 12
#define NORMALS_PER_FACET 3
#define VERTICES_PER_FACET 9

#define TRANSFORM_FLOATS 16

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

int stlWrite(
    const char* stlFile, 
    std::vector<float> &verticies, 
    std::vector<float> &normals);

void VertexTransform(            
            float *xMat,    
            float *vi,      
            float *verto,
						int n)       	      
{
	for(int i = 0; i < n; i+=VERTICES_PER_FACET)
	{
    // do the matTransform               	
                                         	
    // x1-3 = 0 3 6                      	
    // y1-3 = 1 4 7                      	
    // z1-3 = 2 5 8                      	
                                            
    //Vertex Transform 2.0  					
    //Now with a 4x4 matrix!  				
    //x coordinates                                      							
    verto[i+0] = xMat[0]*vi[i+0] + xMat[4]*vi[i+1] + xMat[8]*vi[i+2] + xMat[12]; 	
    verto[i+1] = xMat[0]*vi[i+3] + xMat[4]*vi[i+4] + xMat[8]*vi[i+5] + xMat[12]; 	
    verto[i+2] = xMat[0]*vi[i+6] + xMat[4]*vi[i+7] + xMat[8]*vi[i+8] + xMat[12]; 	
                                                                                 	
    //y coordinates                                                              	
    verto[i+3] = xMat[1]*vi[i+0] + xMat[5]*vi[i+1] + xMat[9]*vi[i+2] + xMat[13]; 	
    verto[i+4] = xMat[1]*vi[i+3] + xMat[5]*vi[i+4] + xMat[9]*vi[i+5] + xMat[13]; 	
    verto[i+5] = xMat[1]*vi[i+6] + xMat[5]*vi[i+7] + xMat[9]*vi[i+8] + xMat[13]; 	
                                                                                 	
    //z coordinates                                                              	
    verto[i+6] = xMat[2]*vi[i+0] + xMat[6]*vi[i+1] + xMat[10]*vi[i+2] + xMat[14];	
    verto[i+7] = xMat[2]*vi[i+3] + xMat[6]*vi[i+4] + xMat[10]*vi[i+5] + xMat[14];	
    verto[i+8] = xMat[2]*vi[i+6] + xMat[6]*vi[i+7] + xMat[10]*vi[i+8] + xMat[14];	
	}
}                                                                                	


void ComputeNormals(                             
            float *vi,                    
            float *no,
						int n)
{
	int ii ;         
	int io ;         

	for(int ii = 0; ii < n; ii+=NORMALS_PER_FACET)
	{
		io =ii*3;
	
		//fairly sure this actually works              
		float t[4];                                    
		t[0] = (vi[ii+4]-vi[ii+3])*(vi[ii+8]-vi[ii+6]) - (vi[ii+7]-vi[ii+6])*(vi[ii+5]-vi[ii+3]); 			
		t[1] = (vi[ii+7]-vi[ii+6])*(vi[ii+2]-vi[ii+0]) - (vi[ii+1]-vi[ii+0])*(vi[ii+8]-vi[ii+6]); 			
		t[2] = (vi[ii+1]-vi[ii+0])*(vi[ii+5]-vi[ii+3]) - (vi[ii+4]-vi[ii+3])*(vi[ii+2]-vi[ii+0]); 			
		t[3] = t[1]+t[2]+t[3];                         
																									 
		no[io  ] = t[0]/t[3];                           
		no[io+1] = t[1]/t[3];                          
		no[io+2] = t[2]/t[3];                          
	}
}                                                  


//float stlVerifyTransform(
//    const float* xMat, 
//    float* v, 
//    float* xformedv, 
//    unsigned int nFaces)
//{
//    float verto[9];
//
//    for( size_t i = 0; i < nFaces; i += 12)
//    {
//        // x1-3 = 0 1 2
//        // y1-3 = 3 4 5
//        // z1-3 = 6 7 8
//
//        //x coordinates
//        verto[i+0] = xMat[0]*v[i+0] + xMat[1]*v[i+3] + xMat[2]*v[i+6] + xMat[3];
//        verto[i+1] = xMat[0]*v[i+1] + xMat[1]*v[i+4] + xMat[2]*v[i+7] + xMat[3];
//        verto[i+2] = xMat[0]*v[i+2] + xMat[1]*v[i+5] + xMat[2]*v[i+8] + xMat[3];
//                        
//        //y coordinates 
//        verto[i+3] = xMat[4]*v[i+0] + xMat[5]*v[i+3] + xMat[6]*v[i+6] + xMat[7];
//        verto[i+4] = xMat[4]*v[i+1] + xMat[5]*v[i+4] + xMat[6]*v[i+7] + xMat[7];
//        verto[i+5] = xMat[4]*v[i+2] + xMat[5]*v[i+5] + xMat[6]*v[i+8] + xMat[7];
//                       
//        //z coordinates
//        verto[i+6] = xMat[8]*v[i+0] + xMat[9]*v[i+3] + xMat[10]*v[i+6] + xMat[11];
//        verto[i+7] = xMat[8]*v[i+1] + xMat[9]*v[i+4] + xMat[10]*v[i+7] + xMat[11];
//        verto[i+8] = xMat[8]*v[i+2] + xMat[9]*v[i+5] + xMat[10]*v[i+8] + xMat[11];
//
//        for(int j = 0; j < 9; ++j)
//            if(fabs(verto[j+i]) - fabs(xformedv[j+i]) > 1e-4 )
//                return j+i;
//    }
//    return 0.0;
//}
