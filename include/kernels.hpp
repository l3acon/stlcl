#ifndef KERNELS_H
#define KERNELS_H

typedef struct vertex_t 
{
  float x1, y1, z1;
  float x2, y2, z2;
  float x3, y3, z3;
} Vertex;


// this kernel takes ~40-50 FLOPS
// Global memory reads: 12*4 (xMat) + 9*4 (vi) = 84 BYTES
// Global memory writes: 9*4 (verto) = 36 BYTES

// (GDDR5 can transfer at most 32 BYTES per clock)

const char * stl_cl_vertexTransform_kernel_source  =
"__kernel                               "
"\nvoid _kVertexTransform(            	"
"\n            __global float *xMat,    "
"\n            __global float *vi,      "
"\n            __global float *verto)       	"      //restrict?
"\n{                                        	"
"\n    // Get the work-item’s unique ID     	"
"\n    unsigned int i = 9*get_global_id(0); 	"
"\n                                         	"
"\n    // do the matTransform               	"
"\n                                         	"
"\n    // x1-3 = 0 1 2                      	"
"\n    // y1-3 = 3 4 5                      	"
"\n    // z1-3 = 6 7 8                      	"
"\n                                             "
"\n    //there's a decent chance this doesnt actually work  							"
"\n    //x coordinates                                      							"
"\n    verto[i+0] = xMat[0]*vi[i+0] + xMat[1]*vi[i+3] + xMat[2]*vi[i+6] + xMat[3];      "
"\n    verto[i+1] = xMat[0]*vi[i+1] + xMat[1]*vi[i+4] + xMat[2]*vi[i+7] + xMat[3];      "
"\n    verto[i+2] = xMat[0]*vi[i+2] + xMat[1]*vi[i+5] + xMat[2]*vi[i+8] + xMat[3];      "
"\n                                                                                     "
"\n    //y coordinates                                                                  "
"\n    verto[i+3] = xMat[4]*vi[i+0] + xMat[5]*vi[i+3] + xMat[6]*vi[i+6] + xMat[7];      "
"\n    verto[i+4] = xMat[4]*vi[i+1] + xMat[5]*vi[i+4] + xMat[6]*vi[i+7] + xMat[7];      "
"\n    verto[i+5] = xMat[4]*vi[i+2] + xMat[5]*vi[i+5] + xMat[6]*vi[i+8] + xMat[7];      "
"\n                                                                                     "
"\n    //z coordinates                                                                  "
"\n    verto[i+6] = xMat[8]*vi[i+0] + xMat[9]*vi[i+3] + xMat[10]*vi[i+6] + xMat[11];    "
"\n    verto[i+7] = xMat[8]*vi[i+1] + xMat[9]*vi[i+4] + xMat[10]*vi[i+7] + xMat[11];    "
"\n    verto[i+8] = xMat[8]*vi[i+2] + xMat[9]*vi[i+5] + xMat[10]*vi[i+8] + xMat[11];    "
"\n                                                                                     "
"\n}                                                                                    "
;


// this kernel takes ~16 FLOPS
// Global memory reads: 9*4 (vi) = 36 BYTES
// Global memory writes: 3*4 (verto) = 12 BYTES

// (GDDR5 can transfer at most 32 BYTES per clock)

const char * stl_cl_computeNormal_kernel_source  =
"__kernel                                               "
"\nvoid _kComputeNormal(                                "
"\n            __global float *vi,                      "
"\n            __global float *no)                   	"
"\n{                                                    "
"\n                                                     "
"\n    // Get the work-item’s unique ID                 "
"\n    unsigned int ii = 9*get_global_id(0);            "
"\n    unsigned int io = 3*get_global_id(0);            "
"\n                                                     "
"\n    //fairly sure this actually works                "
"\n    float t[4];                                      "
"\n    t[0] = (vi[ii+4]-vi[ii+3])*(vi[ii+8]-vi[ii+6])   "
" - (vi[ii+7]-vi[ii+6])*(vi[ii+5]-vi[ii+3]); 			"
"\n    t[1] = (vi[ii+7]-vi[ii+6])*(vi[ii+2]-vi[ii+0])   "
" - (vi[ii+1]-vi[ii+0])*(vi[ii+8]-vi[ii+6]); 			"
"\n    t[2] = (vi[ii+1]-vi[ii+0])*(vi[ii+5]-vi[ii+3])   "
" - (vi[ii+4]-vi[ii+3])*(vi[ii+2]-vi[ii+0]); 			"
"\n    t[3] = t[1]+t[2]+t[3];                           "
"\n                                                     "
"\n    no[io  ] = t[0]/t[3];                         	"   
"\n    no[io+1] = t[1]/t[3];                         	"  
"\n    no[io+2] = t[2]/t[3];                         	"  
"\n}                                                    "
;

// 	this kernel does no FLOPs (only integer 
// 	arithmetic to compute indicies)
//	
//	this kernel access two global elements
// 	in order to sort a data set 
//	this kernel needs to be 
//	called about lg(n)*lg(n) times
// 	probably like n * lg^2(n) memory accessess
//	for each block of data (in this case
// 	each 9 * 4 = 36 BYTE block)

const char * bitonic_STL_sort_source  =
"\n typedef struct vertex_t 	"
"\n {							"
"\n  float x1, y1, z1;			"
"\n  float x2, y2, z2;			"
"\n  float x3, y3, z3;			"
"\n } Vertex;					"
"\n 							"
"__kernel                       "
"\n void _kBitonic_STL_Sort(                    "
"\n             __global Vertex *input_ptr,     "
"\n             const unsigned int stage,       "
"\n             const int passOfStage)          "
"\n {                                    		"
"\n                                      		"
"\n      unsigned int  threadId = get_global_id(0);  				"
"\n      unsigned int  pairDistance = 1 << (stage - passOfStage);   "
"\n      unsigned int  blockWidth = 2 * pairDistance;    			"
"\n      unsigned int  temp;  										"
"\n                                                          		"
"\n      int compareResult;                                      	"
"\n      unsigned int  leftId = (threadId & (pairDistance - 1)) + 	" 
"			(threadId >> (stage - passOfStage) ) * blockWidth;  	" 
"\n      unsigned int  rightId = leftId + pairDistance;  			" 
"\n        															" 
"\n      Vertex leftElement, rightElement;  	" 
"\n    	 Vertex *greater, *lesser;  			" 
"\n 											"
"\n		leftElement.x1 = input_ptr[leftId].x1;	"
"\n		leftElement.y1 = input_ptr[leftId].y1;	"
"\n		leftElement.z1 = input_ptr[leftId].z1;	"
"\n		leftElement.x2 = input_ptr[leftId].x2;	"
"\n		leftElement.y2 = input_ptr[leftId].y2;	"
"\n		leftElement.z2 = input_ptr[leftId].z2;	"
"\n		leftElement.x3 = input_ptr[leftId].x3;	"
"\n		leftElement.y3 = input_ptr[leftId].y3;	"
"\n		leftElement.z3 = input_ptr[leftId].z3;	"
"\n 	rightElement.x1 = input_ptr[rightId].x1;  "
"\n 	rightElement.y1 = input_ptr[rightId].y1;  "
"\n 	rightElement.z1 = input_ptr[rightId].z1;  "
"\n 	rightElement.x2 = input_ptr[rightId].x2;  "
"\n 	rightElement.y2 = input_ptr[rightId].y2;  "
"\n 	rightElement.z2 = input_ptr[rightId].z2;  "
"\n 	rightElement.x3 = input_ptr[rightId].x3;  "
"\n 	rightElement.y3 = input_ptr[rightId].y3;  "
"\n 	rightElement.z3 = input_ptr[rightId].z3;  "
"\n      																" 
"\n      unsigned int sameDirectionBlockWidth = threadId >> stage;   	" 
"\n      unsigned int sameDirection = sameDirectionBlockWidth & 0x1; 	" 
"\n      																" 
"\n      temp = sameDirection ? rightId : temp; 						" 
"\n      rightId = sameDirection ? leftId : rightId; 					" 
"\n      leftId = sameDirection ? temp : leftId;						" 
"\n       																" 
"\n      compareResult = (leftElement.z1 < rightElement.z1); 			" 
"\n       																" 
"\n      greater = compareResult ? &rightElement : &leftElement; 		" 
"\n      lesser = compareResult ? &leftElement : &rightElement; 		" 
"\n       																" 
"\n 	input_ptr[leftId].x1 = lesser->x1;   "
"\n 	input_ptr[leftId].y1 = lesser->y1;   "
"\n 	input_ptr[leftId].z1 = lesser->z1;   "
"\n 	input_ptr[leftId].x2 = lesser->x2;   "
"\n 	input_ptr[leftId].y2 = lesser->y2;   "
"\n 	input_ptr[leftId].z2 = lesser->z2;   "
"\n 	input_ptr[leftId].x3 = lesser->x3;   "
"\n 	input_ptr[leftId].y3 = lesser->y3;   "
"\n 	input_ptr[leftId].z3 = lesser->z3;   "
"\n 	input_ptr[rightId].x1 = greater->x1;  	"
"\n 	input_ptr[rightId].y1 = greater->y1;  	"
"\n 	input_ptr[rightId].z1 = greater->z1;  	"
"\n 	input_ptr[rightId].x2 = greater->x2;  	"
"\n 	input_ptr[rightId].y2 = greater->y2;  	"
"\n 	input_ptr[rightId].z2 = greater->z2;  	"
"\n 	input_ptr[rightId].x3 = greater->x3;  	"
"\n 	input_ptr[rightId].y3 = greater->y3;  	"
"\n 	input_ptr[rightId].z3 = greater->z3;  	"
"\n }     									    " 
; 


#endif

