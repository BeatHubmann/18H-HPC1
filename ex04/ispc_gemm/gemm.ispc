// File       : gemm.ispc
// Created    : Wed Oct 17 2018 09:43:19 AM (+0200)
// Description: ISPC GEMM body
// Copyright 2018 ETH Zurich. All Rights Reserved.
#include "common.h"

/**
 * @brief General matrix-matrix multiplication kernel (GEMM). Computes C = AB.
 * ISPC implementation used for SSE2 and AVX2 extended ISA
 *
 * @param A Matrix dimension p x r
 * @param B Matrix dimension r x q
 * @param C Matrix dimension p x q
 * @param p Dimensional parameter
 * @param r Dimensional parameter
 * @param q Dimensional parameter
 */
/* TODO: Missing piece */
export 
#ifdef _ISPC_SSE2_
void gemm_sse2(
#else
void gemm_avx2(
#endif /* _ISPC_SSE2_ */
        const uniform Real* const uniform A,
        const uniform Real* const uniform B,
              uniform Real* const uniform C,
        const uniform int p,
        const uniform int r,
        const uniform int q)
{
    ///////////////////////////////////////////////////////////////////////////
    // TODO: Write your ISPC implementation of a matrix-matrix multiplication
    // here.  Try to use a general type for floating point numbers, such that
    // you can easily compile this kernel for float (single precision, 32bit)
    // and double (double precision, 64bit).  Check out the 'common.h' header
    // for a hint.  Note: A working code can be implemented with ~15 lines of
    // ISPC code.
    //
    // We use two different function names for the same function, such that we
    // can compile for two different targets (SSE2 and AVX2) and link both of
    // them to our application code in 'gemm.cpp'.
    ///////////////////////////////////////////////////////////////////////////
	

	Real* B_t= new Real[r * q];
	foreach (i= 0 ... q, j= 0 ... r)
		B_t[i + j * r]= B[i * q + j];	
	 
    foreach (i= 0 ... p, j= 0 ... q)
    {
        Real element_sum= 0;
        for (int k= 0; k < r; k++)
            element_sum += A[i * p + k] * B_t[k + j * q];
        C[i * p + j]= element_sum;
    }
}
