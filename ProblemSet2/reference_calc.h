#ifndef REFERENCE_H__
#define REFERENCE_H__

#include <cuda_runtime.h>
void referenceCalculation(const uchar4* const rgbaImage, uchar4* const outputImage, size_t numRows, size_t numCols,
                          const float* const filter, const int filterWidth);

#endif