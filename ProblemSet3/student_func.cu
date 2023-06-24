/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/
#include "utils.h"

__global__ void find_min_max(const float* const vals, float* res, bool findMin, int size) {
  extern __shared__ float sdata[];
  unsigned long id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < size) {
    sdata[threadIdx.x] = vals[id];
  } else {
    sdata[threadIdx.x] = vals[0];
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i = i / 2) {
    if (threadIdx.x < i) {
      if (findMin) {
        sdata[threadIdx.x] = std::fmin(sdata[threadIdx.x], sdata[threadIdx.x + i]);
      } else {
        sdata[threadIdx.x] = std::fmax(sdata[threadIdx.x], sdata[threadIdx.x + i]);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    res[blockIdx.x] = sdata[0];
  }
  return;
}

__global__ void compute_histogram(const float* const vals, unsigned int* const d_hist, float bucketWidth,
                                  float min, unsigned int size) {
  unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id > size) {
    return;
  }
  float val = vals[id];
  unsigned int bucketId = (unsigned int)((val - min) / bucketWidth);
  atomicAdd(&d_hist[bucketId], 1);
}

__global__ void scan_hist(unsigned int* const d_vals, int size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= size) {
    return;
  }
  for (int i = 1; i < size; i *= 2) {
    int old = 0;
    if (id - i >= 0) {
      old = d_vals[id - i];
    }
    __syncthreads();
    if (id - i >= 0) {
      d_vals[i] += old;
    }
    __syncthreads();
  }
  if (id == 0) {
    for (int i = size - 1; i > 0; i--) {
      d_vals[i] = d_vals[i - 1];
    }
    d_vals[0] = 0;
  }
}

float findMinMax(const float* const d_logLuminance, unsigned int blockLen, bool findMin, unsigned int size) {
  float *d_vals, *d_res;
  unsigned int gridLen = ((size) + (blockLen - 1)) / blockLen;
  size_t vals_size = sizeof(float) * size;
  checkCudaErrors(cudaMalloc(&d_vals, vals_size));
  checkCudaErrors(cudaMemcpy(d_vals, d_logLuminance, vals_size, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMalloc(&d_res, sizeof(float) * gridLen));
  int len = size;
  while (true) {
    find_min_max<<<gridLen, blockLen, blockLen * sizeof(float)>>>(d_vals, d_res, findMin, len);
    checkCudaErrors(cudaMemcpy(d_vals, d_res, sizeof(float) * gridLen, cudaMemcpyDeviceToDevice));
    if (gridLen == 1) {
      break;
    }
    gridLen = (gridLen + blockLen - 1) / blockLen;
    len = (len + blockLen - 1) / blockLen;
  }
  float h_res;
  checkCudaErrors(cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_vals));
  checkCudaErrors(cudaFree(d_res));
  return h_res;
}

#define TEST_ARR_LEN 100 * 1024
void test_minmax() {
  float* d_arr;
  checkCudaErrors(cudaMalloc(&d_arr, sizeof(float) * TEST_ARR_LEN));
  float* h_arr = (float*)malloc(TEST_ARR_LEN * sizeof(float));
  h_arr[0] = 0.0;
  for (int i = 1; i < TEST_ARR_LEN; i++) {
    h_arr[i] = h_arr[i - 1] + 0.0001;
  }
  checkCudaErrors(cudaMemcpy(d_arr, h_arr, TEST_ARR_LEN * sizeof(float), cudaMemcpyHostToDevice));
  size_t blockLen = 1024;
  cudaDeviceSynchronize();
  float min_logLum = findMinMax(d_arr, blockLen, true, TEST_ARR_LEN);
  float max_logLum = findMinMax(d_arr, blockLen, false, TEST_ARR_LEN);
  std::cout << "TEST: " << min_logLum << ", " << max_logLum << std::endl;
  checkCudaErrors(cudaFree(d_arr));
  free(h_arr);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance, unsigned int* const d_cdf,
                                  float& min_logLum, float& max_logLum, const size_t numRows,
                                  const size_t numCols, const size_t numBins) {
  // TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  unsigned int blockLen = 1024;
  min_logLum = findMinMax(d_logLuminance, blockLen, true, numRows * numCols);
  max_logLum = findMinMax(d_logLuminance, blockLen, false, numRows * numCols);
  std::cout << min_logLum << ", " << max_logLum << std::endl;
  float range = max_logLum - min_logLum;
  float bucketWidth = range / numBins;
  unsigned int gridLen = ((numRows * numCols) + (blockLen - 1)) / blockLen;
  compute_histogram<<<gridLen, blockLen>>>(d_logLuminance, d_cdf, bucketWidth, min_logLum, numRows * numCols);
  scan_hist<<<1, numBins>>>(d_cdf, numBins);
  std::cout << "calling test " << std::endl;
  test_minmax();
}
