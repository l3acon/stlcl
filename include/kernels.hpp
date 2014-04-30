#ifndef KERNELS_H
#define KERNELS_H

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
"\n    // x1-3 = 0 3 6                      	"
"\n    // y1-3 = 1 4 7                      	"
"\n    // z1-3 = 2 5 8                      	"
"\n                                             "
"\n    //Vertex Transform 2.0  					"
"\n    //Now with a 4x4 matrix!  				"
"\n    verto[i+0] = xMat[0]*vi[i+0] + xMat[4]*vi[i+1] + xMat[8]* vi[i+2] + xMat[12];	"
"\n    verto[i+1] = xMat[1]*vi[i+0] + xMat[5]*vi[i+1] + xMat[9]* vi[i+2] + xMat[13];	"
"\n    verto[i+2] = xMat[2]*vi[i+0] + xMat[6]*vi[i+1] + xMat[10]*vi[i+2] + xMat[14];	"
"\n 																				"
"\n    verto[i+3] = xMat[0]*vi[i+3] + xMat[4]*vi[i+4] + xMat[8]*vi [i+5] + xMat[12];	"
"\n    verto[i+4] = xMat[1]*vi[i+3] + xMat[5]*vi[i+4] + xMat[9]*vi [i+5] + xMat[13];	"	
"\n    verto[i+5] = xMat[2]*vi[i+3] + xMat[6]*vi[i+4] + xMat[10]*vi[i+5] + xMat[14];	"	
"\n    																				"
"\n    verto[i+6] = xMat[0]*vi[i+6] + xMat[4]*vi[i+7] + xMat[8]*vi [i+8] + xMat[12];	"	
"\n    verto[i+7] = xMat[1]*vi[i+6] + xMat[5]*vi[i+7] + xMat[9]*vi [i+8] + xMat[13];	"
"\n    verto[i+8] = xMat[2]*vi[i+6] + xMat[6]*vi[i+7] + xMat[10]*vi[i+8] + xMat[14];	"	
"\n                                                                                 	"
"\n}                                                                                	"
;


// this kernel takes ~16 FLOPS
// Global memory reads: 9*4 (vi) = 36 BYTES
// Global memory writes: 3*4 (verto) = 12 BYTES

// (GDDR5 can transfer at most 32 BYTES per clock)

const char * stl_cl_computeNormals_kernel_source  =
"__kernel                                               "
"\nvoid _kComputeNormals(                                "
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
"\n    t[3] = t[0]+t[1]+t[2];                           "
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
"__kernel                                           "
"\n void _kBitonic_STL_Sort(                    	"
"\n             __global float *input_ptr,        	"
"\n             const unsigned int stage,           "
"\n             const int passOfStage)              "
"\n {                                    			"
"\n                                      			"
"\n      unsigned int  threadId = get_global_id(0);  					"
"\n      unsigned int  pairDistance = 1 << (stage - passOfStage);   	"
"\n      unsigned int  blockWidth = 2 * pairDistance;    				"
"\n      unsigned int  temp;  											"
"\n                                                          			"
"\n      int compareResult;                                      		"
"\n      unsigned int  leftId = (threadId & (pairDistance - 1)) + 		" 
"			(threadId >> (stage - passOfStage) ) * blockWidth;  		" 
"\n      unsigned int  rightId = leftId + pairDistance;  				" 
"\n        																" 
"\n      float leftElement[9];  							" 
"\n 	 float rightElement[9];								"
"\n    	 float *greater, *lesser;  						" 
"\n 													"
"\n		leftElement[0] = input_ptr[leftId*9 + 0];	"
"\n		leftElement[1] = input_ptr[leftId*9 + 1];	"
"\n		leftElement[2] = input_ptr[leftId*9 + 2];	"
"\n		leftElement[3] = input_ptr[leftId*9 + 3];	"
"\n		leftElement[4] = input_ptr[leftId*9 + 4];	"
"\n		leftElement[5] = input_ptr[leftId*9 + 5];	"
"\n		leftElement[6] = input_ptr[leftId*9 + 6];	"
"\n		leftElement[7] = input_ptr[leftId*9 + 7];	"
"\n		leftElement[8] = input_ptr[leftId*9 + 8];	"
"\n		rightElement[0] = input_ptr[rightId*9 + 0];	"
"\n		rightElement[1] = input_ptr[rightId*9 + 1];	"
"\n		rightElement[2] = input_ptr[rightId*9 + 2];	"
"\n		rightElement[3] = input_ptr[rightId*9 + 3];	"
"\n		rightElement[4] = input_ptr[rightId*9 + 4];	"
"\n		rightElement[5] = input_ptr[rightId*9 + 5];	"
"\n		rightElement[6] = input_ptr[rightId*9 + 6];	"
"\n		rightElement[7] = input_ptr[rightId*9 + 7];	"
"\n		rightElement[8] = input_ptr[rightId*9 + 8];	"
"\n 												"
"\n      unsigned int sameDirectionBlockWidth = threadId >> stage;   	" 
"\n      unsigned int sameDirection = sameDirectionBlockWidth & 0x1; 	" 
"\n      																" 
"\n      temp = sameDirection ? rightId : temp; 						" 
"\n      rightId = sameDirection ? leftId : rightId; 					" 
"\n      leftId = sameDirection ? temp : leftId;						" 
"\n       																" 
"\n      compareResult = (leftElement[2] < rightElement[2]); 			" 
"\n       																" 
"\n      greater = compareResult ? rightElement : leftElement; 			" 
"\n      lesser = compareResult ? leftElement : rightElement; 			" 
"\n       																" 
"\n 	input_ptr[leftId*9 + 0] = lesser[0];   "
"\n 	input_ptr[leftId*9 + 1] = lesser[1];   "
"\n 	input_ptr[leftId*9 + 2] = lesser[2];   "
"\n 	input_ptr[leftId*9 + 3] = lesser[3];   "
"\n 	input_ptr[leftId*9 + 4] = lesser[4];   "
"\n 	input_ptr[leftId*9 + 5] = lesser[5];   "
"\n 	input_ptr[leftId*9 + 6] = lesser[6];   "
"\n 	input_ptr[leftId*9 + 7] = lesser[7];   "
"\n 	input_ptr[leftId*9 + 8] = lesser[8];   "
"\n 	input_ptr[rightId*9 + 0] = greater[0];   "
"\n 	input_ptr[rightId*9 + 1] = greater[1];   "
"\n 	input_ptr[rightId*9 + 2] = greater[2];   "
"\n 	input_ptr[rightId*9 + 3] = greater[3];   "
"\n 	input_ptr[rightId*9 + 4] = greater[4];   "
"\n 	input_ptr[rightId*9 + 5] = greater[5];   "
"\n 	input_ptr[rightId*9 + 6] = greater[6];   "
"\n 	input_ptr[rightId*9 + 7] = greater[7];   "
"\n 	input_ptr[rightId*9 + 8] = greater[8];   "
"\n }     										" 
; 

#endif

