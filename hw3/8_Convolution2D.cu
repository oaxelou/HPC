/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

// #include <cuda.h>
// #include <cuda_runtime_api.h>

#include <time.h>
#include "gputimer.h"

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define PADDING filter_radius
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005



////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter,
                       int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = PADDING; (unsigned int)y < (unsigned int)imageH - PADDING; y++) {
    for (x = PADDING; (unsigned int)x < (unsigned int)imageW - PADDING; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }

        h_Dst[y * imageW + x] = sum;
      }
    }
  }

}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = PADDING; (unsigned int)y < (unsigned int)imageH - PADDING; y++) {
    for (x = PADDING; (unsigned int)x < (unsigned int)imageW - PADDING; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }

        h_Dst[y * imageW + x] = sum;
      }
    }
  }

}


////////////////////////////////////////////////////////////////////////////////
// GPU: Row convolution Kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_rows(const float *filter, const float *input, float *output,
                       int imageW, int imageH, int filterR){
  int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
  int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

  int padding = filterR;

  float sum = 0;
  int k;

  // Rows
  for(k = -filterR; k <= filterR; k++){
    int d = (idx_x + padding) + k;

    sum += input[(idx_y + padding) * imageW + d] * filter[filterR - k];
  }
  output[(idx_y + padding) * imageW + idx_x + padding] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// GPU: Column convolution Kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_columns(const float *filter, const float *buffer, float *output,
                       int imageW, int imageH, int filterR){
  int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
  int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

  int padding = filterR;

  float sum = 0;
  int k;

  // Columns
  for(k = -filterR; k <= filterR; k++){
    int d = (idx_y + padding) + k;

    sum += buffer[d * imageW + (idx_x + padding)] * filter[filterR - k];
  }
  output[(idx_y + padding) * imageW + idx_x + padding] = sum;
}

// Auxiliary function for CUDA error checking
void cudaCheckForErrors(){
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess){
    // something's gone wrong
    // print out the CUDA error as a string
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    exit(1);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(void) {

    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU;

    // GPU
    float *d_Filter, *d_Input, *d_Buffer, *d_OutputGPU, *h_OutputGPU;

    unsigned int imageW;
    unsigned int imageH;
    unsigned int i;
    unsigned int j;

    unsigned int padding_imageW;
    unsigned int padding_imageH;

    GpuTimer timer;
    clock_t start_CPU, end_CPU;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    padding_imageH = PADDING * 2 + imageH;
    padding_imageW = padding_imageH;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(padding_imageW * padding_imageH * sizeof(float));
    h_Buffer    = (float *)malloc(padding_imageW * padding_imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(padding_imageW * padding_imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(padding_imageW * padding_imageH * sizeof(float));


    cudaMalloc( (void **) &d_Filter,      FILTER_LENGTH * sizeof(float));
    cudaMalloc( (void **) &d_Input, padding_imageW * padding_imageH * sizeof(float));
    cudaMalloc( (void **) &d_Buffer,padding_imageW * padding_imageH * sizeof(float));
    cudaMalloc( (void **) &d_OutputGPU, padding_imageW * padding_imageH * sizeof(float));

    if(!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU || !h_OutputGPU){
        printf("error allocating memory for the host\n");
        exit(1);
   }

    if(!d_Filter || !d_Input || !d_Buffer || !d_OutputGPU){
      printf("Error allocating memory for the device\n");
      exit(1);
    }

    cudaMemset(d_Buffer, 0, padding_imageW * padding_imageW * sizeof(float));
    cudaMemset(d_OutputGPU, 0, padding_imageW * padding_imageW * sizeof(float));

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < padding_imageH; i++) {
      for (j = 0; j < padding_imageW; j++) {
        if(i < PADDING || i >= imageW + PADDING || j < PADDING || j >= imageW + PADDING)
          h_Input[i * padding_imageW + j] = 0;
        else
          h_Input[i*padding_imageW + j] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
      }
    }

    //////////////////////////////// CPU ///////////////////////////////////////
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    start_CPU = clock();
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, padding_imageW, padding_imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, padding_imageW, padding_imageH, filter_radius); // convolution kata sthles

    end_CPU = clock();
    printf("CPU Time: %lf ms\n", ((double) ((end_CPU - start_CPU) * 1000)) / CLOCKS_PER_SEC);

    //////////////////////////////// GPU ///////////////////////////////////////
    dim3 block_dim;
    dim3 grid_dim;

    if(padding_imageW < 32){
      block_dim.x = imageH;
      block_dim.y = imageH;

      grid_dim.x = 1;
      grid_dim.y = 1;

    } else{
      block_dim.x = 32;
      block_dim.y = 32;

      grid_dim.x = imageW / block_dim.x;
      grid_dim.y = imageH / block_dim.y;
}

    printf("GPU computation...\n");

    timer.Start();

    //cudaMemcpy(d_Buffer, h_Input, padding_imageW * padding_imageW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, padding_imageW * padding_imageW * sizeof(float), cudaMemcpyHostToDevice);


    kernel_rows<<<grid_dim, block_dim>>>(d_Filter, d_Input, d_Buffer, padding_imageW, padding_imageH, filter_radius);
    cudaDeviceSynchronize();
    cudaCheckForErrors();

    kernel_columns<<<grid_dim, block_dim>>>(d_Filter, d_Buffer, d_OutputGPU, padding_imageW, padding_imageH, filter_radius);
    cudaDeviceSynchronize();
    cudaCheckForErrors();

    cudaMemcpy(h_OutputGPU, d_OutputGPU, padding_imageW * padding_imageH * sizeof(float), cudaMemcpyDeviceToHost);

    timer.Stop();
    printf("GPU Time elapsed = %g ms\n", timer.Elapsed());

    //////////////////////// RESULT COMPARISON /////////////////////////////////

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas

    for(i = 0; i < imageH * imageW; i++){
      if(ABS(h_OutputGPU[i] - h_OutputCPU[i]) >= accuracy){
        printf("GPU computations are not as accurate as we want.\n");
        break;
      }
    }

    ////////////////// CPU: free all the allocated memory //////////////////////
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    ////////////////// GPU: free all the allocated memory //////////////////////
    free(h_OutputGPU);
    cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_Buffer);
    cudaFree(d_OutputGPU);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
