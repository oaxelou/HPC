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

#define BLOCK_SIZE 2048

// #define CPU_CODE
////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter,
                       int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = PADDING; (unsigned int)y < (unsigned int)imageH - PADDING; y++) {
    for (x = PADDING; (unsigned int)x < (unsigned int)imageW - PADDING; x++) {
      double sum = 0;

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
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = PADDING; (unsigned int)y < (unsigned int)imageH - PADDING; y++) {
    for (x = PADDING; (unsigned int)x < (unsigned int)imageW - PADDING; x++) {
      double sum = 0;

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
__global__ void kernel_rows(const double *filter, const double *input, double *output,
                       int imageW, int imageH, int filterR){
  int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
  int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

  int padding = filterR;

  double sum = 0;
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
__global__ void kernel_columns(const double *filter, const double *buffer, double *output,
                       int imageW, int imageH, int filterR){
  int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
  int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

  int padding = filterR;

  double sum = 0;
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

    double
    *h_Filter,
    *h_Input,
    *h_Input_block_size,
    *h_Buffer,
    *h_Buffer_block_size,
    *h_OutputGPU_block_size;

#ifdef CPU_CODE
    double *h_OutputCPU;
#endif
    // GPU
    double *d_Filter, *d_Input, *d_Buffer, *d_OutputGPU, *h_OutputGPU;

    unsigned int imageW;
    unsigned int imageH;
    unsigned int i;
    unsigned int j;

    unsigned int padding_imageW;
    unsigned int padding_imageH;

    GpuTimer timer;
#ifdef CPU_CODE
    clock_t start_CPU, end_CPU;
#endif

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
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)calloc(padding_imageW * padding_imageH, sizeof(double));
    h_Buffer    = (double *)calloc(padding_imageW * padding_imageH, sizeof(double));
    h_OutputGPU = (double *)calloc(padding_imageW * padding_imageH, sizeof(double));

    if(!h_Filter || !h_Input || !h_Buffer || !h_OutputGPU){
      printf("error allocating memory for the host\n");
      exit(1);
    }

#ifdef CPU_CODE
    h_OutputCPU = (double *)calloc(padding_imageW * padding_imageH, sizeof(double));
    if(!h_OutputCPU){
      printf("error allocating memory for h_OutputCPU\n");
      exit(1);
    }
#endif

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < padding_imageH; i++) {
      for (j = 0; j < padding_imageW; j++) {
        if(i < PADDING || i >= imageW + PADDING || j < PADDING || j >= imageW + PADDING)
          h_Input[i * padding_imageW + j] = 0;
        else
          h_Input[i*padding_imageW + j] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
      }
    }

    //////////////////////////////// CPU ///////////////////////////////////////
#ifdef CPU_CODE
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    start_CPU = clock();
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, padding_imageW, padding_imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, padding_imageW, padding_imageH, filter_radius); // convolution kata sthles
    // for(i = 0; i < padding_imageH; i++)
    //   for(j = 0; j < padding_imageW; j++)
    //     printf("\tCPU Output: [%d][%d]: %lf\n", i, j, h_Buffer[i * padding_imageW + j]);

    // for(i = 0; i < padding_imageH; i++)
    //   for(j = 0; j < padding_imageW; j++)
    //     printf("\tCPU Input: [%d][%d]: %lf\n", i, j, h_Input[i * padding_imageW + j]);

    end_CPU = clock();
    printf("CPU Time: %lf ms\n", ((double) ((end_CPU - start_CPU) * 1000)) / CLOCKS_PER_SEC);
#endif

    //////////////////////////////// GPU ///////////////////////////////////////
    unsigned int blocks_height, blocks_width;
    blocks_height = blocks_width = BLOCK_SIZE;

    unsigned int blocks_padded_h, blocks_padded_w;
    blocks_padded_h = blocks_height + 2 * filter_radius;
    blocks_padded_w = blocks_width  + 2 * filter_radius;

    h_Input_block_size     = (double *)calloc(blocks_padded_w * blocks_padded_h, sizeof(double));
    h_Buffer_block_size    = (double *)calloc(blocks_padded_w * blocks_padded_h, sizeof(double));
    h_OutputGPU_block_size = (double *)calloc(blocks_padded_w * blocks_padded_h, sizeof(double));

    if(!h_Input_block_size || !h_Buffer_block_size || !h_OutputGPU_block_size){
      printf("Error allocating auxiliary memory for the host\n");
      exit(1);
    }

    cudaMalloc( (void **) &d_Filter,      FILTER_LENGTH * sizeof(double));
    cudaMalloc( (void **) &d_Input, blocks_padded_w * blocks_padded_h * sizeof(double));
    cudaMalloc( (void **) &d_Buffer,blocks_padded_w * blocks_padded_h * sizeof(double));
    cudaMalloc( (void **) &d_OutputGPU, blocks_padded_w * blocks_padded_h * sizeof(double));

    if(!d_Filter || !d_Input || !d_Buffer || !d_OutputGPU){
      printf("Error allocating memory for the device\n");
      exit(1);
    }

    cudaMemset(d_Buffer, 0, blocks_padded_w * blocks_padded_h * sizeof(double));
    cudaMemset(d_OutputGPU, 0, blocks_padded_w * blocks_padded_h * sizeof(double));


    dim3 block_dim;
    dim3 grid_dim;

    if(blocks_width < 32){
      block_dim.x = blocks_width;
      block_dim.y = blocks_height;

      grid_dim.x = 1;
      grid_dim.y = 1;

    } else{
      block_dim.x = 32;
      block_dim.y = 32;

      grid_dim.x = blocks_width  / block_dim.x;
      grid_dim.y = blocks_height / block_dim.y;
}

    printf("GPU computation...\n");

    for(i = 0; i < padding_imageH * padding_imageW; i++) h_Buffer[i] = 0;
    timer.Start();

    ////////////////////////////////////////////////////////////////////////////
    int steps = imageH / blocks_height; // block size is power of 2 and less than image size
    int step_x, step_y;

    printf("steps: %d\n", steps);
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(double), cudaMemcpyHostToDevice);

    for(step_y= 0; step_y < steps; step_y++){
      for(step_x = 0; step_x < steps; step_x++){

        // for(i = 0; i < padding_imageH; i++)
        //   for(j = 0; j < padding_imageW; j++){
        //     // printf("\tBuffer: %d %d [%d][%d]: %lf\n", step_y, step_x, i, j, h_Buffer[i * padding_imageW + j]);
        //   }

        for(i = 0; i < blocks_padded_h; i++)
          for(j = 0; j < blocks_padded_w; j++){
            // printf("\tInput: %d %d %d %d\n", step_y, step_x,i, j);
            h_Input_block_size[i * blocks_padded_w + j] = h_Input[(step_y * blocks_height + i) * padding_imageW + step_x * blocks_width + j];
            // printf("Input: %d %d [%d][%d]: %lf\n", step_y, step_x, i, j, h_Input_block_size[i * blocks_padded_w + j]);
            // if()
              // printf("[%d][%d]: %lf\n", i, j, h_Input_block_size[i * blocks_padded_w + j]);
          }
        // printf("%d %d\n", step_y, step_x);
        cudaMemcpy(d_Input, h_Input_block_size, blocks_padded_w * blocks_padded_h * sizeof(double), cudaMemcpyHostToDevice);
        kernel_rows<<<grid_dim, block_dim>>>(d_Filter, d_Input, d_Buffer, blocks_padded_w, blocks_padded_h, filter_radius);
        cudaMemcpy(h_Buffer_block_size, d_Buffer, blocks_padded_w * blocks_padded_h * sizeof(double), cudaMemcpyDeviceToHost);

        // printf("step: %d %d\n", step_y, step_x);

        for(i = filter_radius; i < blocks_padded_h - filter_radius; i++){
          for(j = filter_radius; j < blocks_padded_w - filter_radius; j++){
            // printf("[%d][%d] ", step_y * blocks_height + i,  + step_x * blocks_width + j);//, h_Buffer_block_size[i * blocks_padded_w + j]);
            // printf("\tBuffer: %d %d [%d][%d]: %lf ", step_y, step_x, i, j, h_Buffer_block_size[i * blocks_padded_w + j]);
            h_Buffer[(step_y * blocks_height + i) * padding_imageW + step_x * blocks_width + j] = h_Buffer_block_size[i * blocks_padded_w + j];
          }
          // printf("\n");
        }
      }
    }

    cudaDeviceSynchronize();
    cudaCheckForErrors();

    ////////////////////////////////////////////////////////////////////////////
    for(step_y= 0; step_y < steps; step_y++){
      for(step_x = 0; step_x < steps; step_x++){

        for(i = 0; i < blocks_padded_h; i++)
          for(j = 0; j < blocks_padded_w; j++){
            h_Buffer_block_size[i * blocks_padded_w + j] = h_Buffer[(step_y * blocks_height + i) * padding_imageW + step_x * blocks_width + j];
            // printf("Buffer: %d %d [%d][%d]: %lf\n", step_y, step_x, i, j, h_Buffer_block_size[i * blocks_padded_w + j]);
          }

        cudaMemcpy(d_Buffer, h_Buffer_block_size, blocks_padded_w * blocks_padded_h * sizeof(double), cudaMemcpyHostToDevice);
        kernel_columns<<<grid_dim, block_dim>>>(d_Filter, d_Buffer, d_OutputGPU, blocks_padded_w, blocks_padded_h, filter_radius);
        cudaMemcpy(h_OutputGPU_block_size, d_OutputGPU, blocks_padded_w * blocks_padded_h * sizeof(double), cudaMemcpyDeviceToHost);

        for(i = filter_radius; i < blocks_padded_h - filter_radius; i++)
          for(j = filter_radius; j < blocks_padded_w - filter_radius; j++){
            h_OutputGPU[(step_y * blocks_height + i) * padding_imageW + step_x * blocks_width + j] = h_OutputGPU_block_size[i * blocks_padded_w + j];
            // printf("[%d][%d]: %lf\n", i, j, h_OutputGPU[(step_y * blocks_height + i) * padding_imageW + step_x * blocks_width + j]);
          }
      }
    }
    // for(i = 0; i < padding_imageH; i++)
    //   for(j = 0; j < padding_imageW; j++)
    //     printf("GPU Output [%d][%d]: %lf\n", i, j, h_OutputGPU[i * padding_imageW + j]);

    cudaDeviceSynchronize();
    cudaCheckForErrors();

    timer.Stop();
    printf("GPU Time elapsed = %g ms\n", timer.Elapsed());

    //////////////////////// RESULT COMPARISON /////////////////////////////////

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas

    // for(i = 0; i < imageH * imageW; i++){
    //   printf("[%d]: %lf vs %lf\n",i, h_OutputGPU[i], h_OutputCPU[i]);
    //   if(ABS(h_OutputGPU[i] - h_OutputCPU[i]) >= accuracy){
    //     printf("GPU computations are not as accurate as we want.\n");
    //     // break;
    //   }
    // }
#ifdef CPU_CODE
    for(i = 0; i < padding_imageH; i++){
      for(j = 0; j < padding_imageW; j++){
        if(i > filter_radius && i < padding_imageH - filter_radius && j > filter_radius && j < padding_imageW - filter_radius){
          if(ABS(h_OutputGPU[i * padding_imageH + j] - h_OutputCPU[i * padding_imageH + j]) >= accuracy){
            printf("[%d][%d]: %lf vs %lf\n",i, j, h_OutputGPU[i * padding_imageH + j], h_OutputCPU[i * padding_imageH + j]);
            printf("GPU computations are not as accurate as we want.\n");
            // break;
          }
        }
      }
    }
    ////////////////// CPU: free all the allocated memory //////////////////////
    free(h_OutputCPU);
#endif
    free(h_Buffer);
    free(h_Buffer_block_size);
    free(h_Input);
    free(h_Input_block_size);
    free(h_Filter);

    ////////////////// GPU: free all the allocated memory //////////////////////
    free(h_OutputGPU);
    free(h_OutputGPU_block_size);
    cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_Buffer);
    cudaFree(d_OutputGPU);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
