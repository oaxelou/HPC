/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005



////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter,
                       int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
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

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
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

__global__ void kernel(const float *filter, const float *input, float *output){
  int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
  int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

  int grid_width = gridDim.x * blockDim.x;
  int idx = grid_width * idx_y + idx_x;

  // printf("I am: idx_y %d + %d * %d = %d / idx_x %d + %d * %d = %d. Generally: %d\n",
  // threadIdx.y, blockDim.y, blockIdx.y, idx_y,
  // threadIdx.x, blockDim.x, blockIdx.x, idx_x, idx);
  output[idx] = input[idx] + 1;
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
    float *d_Filter, *d_Input, *d_OutputGPU, *h_OutputGPU;

    int imageW;
    int imageH;
    unsigned int i;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));


    cudaMalloc( (void **) &d_Filter,      FILTER_LENGTH * sizeof(float));
    cudaMalloc( (void **) &d_Input,     imageW * imageH * sizeof(float));
    cudaMalloc( (void **) &d_OutputGPU, imageW * imageH * sizeof(float));

    if(!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU ||
       !d_Filter || !d_Input || !d_OutputGPU || !h_OutputGPU){
      printf("Error allocating memory.\n");
      exit(1);
    }

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }

    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);

    printf("Input:\n");
    for(int i = 0; i < imageH; i++){
      for(int j = 0; j < imageW; j++)
        printf("%g ", h_Input[i * imageW + j]);
      printf("\n");
    }

    // ********************* CPU *********************
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    // printf("Results CPU:\n");
    // for(int i = 0; i < imageH; i++){
    //   for(int j = 0; j < imageW; j++)
    //     printf("%g ", h_OutputCPU[i * imageW + j]);
    //   printf("\n");
    // }

    //******************* GPU ************************
    // grid_dim, block_dim

    dim3 block_dim;
    block_dim.x = 4;
    block_dim.y = 4;

    dim3 grid_dim;
    grid_dim.x = imageW / block_dim.x;
    grid_dim.y = imageH / block_dim.x;

    // if((imageW * imageH) % block_size) ++grid_size;  // xreiazetai ?!

    kernel<<<grid_dim, block_dim>>>(d_Filter, d_Input, d_OutputGPU);
    cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Results GPU:\n");
    for(int i = 0; i < imageH; i++){
      for(int j = 0; j < imageW; j++)
        printf("%g ", h_OutputGPU[i * imageW + j]);
      printf("\n");
    }

    //************ RESULT COMPARISON *****************

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas

    // for(int i = 0; i < imageH * imageW; i++){
    //   if(ABS(h_OutputGPU[i] - h_OutputCPU[i]) >= accuracy){
    //     printf("ERROR with CUDA. Something went wrong.\n");
    //     break;
    //   }
    // }

    //********** CPU *************
    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    //********** GPU *************
    free(h_OutputGPU);
    cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_OutputGPU);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
