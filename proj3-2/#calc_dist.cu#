/*
 * Proj 3-2 SKELETON
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include "utils.h"

static unsigned int translated_y = 0;


__global__ void flip_horizontal_kernel(float *dst, float *src, int width) 
{
    int X = threadIdx.x + blockIdx.x * blockDim.x;
    int Y = width * blockIdx.y;
    dst[width - 1 -  X + Y] = src[X + Y];

}
// Must run in device
/* Does a horizontal flip of the array arr */
void flip_horizontal(float *dst, float *src, int width, unsigned int thread_block) 
{
    int blocks_per_grid_x = width / thread_block; // since the template is a power of 2
    int blocks_per_grid_y = width;


    // Call flip_horizontal_kernal
    dim3 dim_thread_per_block(thread_block, 1, 1); // name dim_thread_per_block can be anything
    dim3 dim_blocks_per_grid(blocks_per_grid_x, blocks_per_grid_y);  // two dimensional

    flip_horizontal_kernel<<<dim_blocks_per_grid, dim_thread_per_block>>>(dst, src, width);

    cudaThreadSynchronize();
    CUT_CHECK_ERROR("");
}
__global__ void rotate_ccw_90_kernel(float *dst, float *src, int width) {
    
    int X = threadIdx.x + blockIdx.x * blockDim.x ;
    int srcIndex = X + width*blockIdx.y;
    int dstIndex = blockIdx.y + width*(width - X - 1); 
    dst[dstIndex] = src[srcIndex];
}
// Must run in device
/* Rotates the square array ARR by 90 degrees counterclockwise. */
void rotate_ccw_90(float *dst, float *src, int width, unsigned int thread_block) 
{
    int len = width * width;
    int thread_per_block = thread_block;
     
    int blocks_per_grid_x = width / thread_per_block;
    int blocks_per_grid_y = len / width;

    dim3 dim_thread_per_block(thread_per_block, 1, 1);
    dim3 dim_blocks_per_grid(blocks_per_grid_x, blocks_per_grid_y);

    rotate_ccw_90_kernel<<<dim_blocks_per_grid, dim_thread_per_block>>>(dst, src, width);

    cudaThreadSynchronize();
    CUT_CHECK_ERROR("");
}
__global__ void finding_distance_kernel(float *image, float *temp, int t_width, float *difference, int quantum_y) 
{
    int X = threadIdx.x + blockIdx.x * blockDim.x;
    int image_index = X + blockIdx.y * quantum_y;
    int template_index = X + blockIdx.y * t_width;
    float x_val = image[image_index];
    float y_val = temp[template_index];
    difference[template_index] = (x_val - y_val) * (x_val - y_val);
}

__global__ void reduction_kernel(float *difference, int level, int t_width) 
{
    int thisThreadIndex = (blockIdx.x*blockDim.x + threadIdx.x) * 2 * level;
    int nextIndex = thisThreadIndex + level;
    difference[thisThreadIndex] = difference[thisThreadIndex] + difference[nextIndex];
}


// Must run in device
float finding_distance(float *image, float *temp, int t_width, float *difference) 
{ // difference is 0-0

    //float result = 0;
    unsigned int len = t_width * t_width;
    unsigned int level = 1;
    unsigned int thread_per_block;
    if (t_width < 512)
    { 
	thread_per_block = t_width;
    }
    else 
    {
	thread_per_block = 512;
    }

    unsigned int blocks_per_grid_x = t_width / thread_per_block;
    unsigned int blocks_per_grid_y = t_width; // it is same as t_width;
    unsigned int quantum_y = translated_y ? translated_y : t_width;

    dim3 dim_blocks_per_grid(blocks_per_grid_x, blocks_per_grid_y);
    dim3 dim_thread_per_block(thread_per_block, 1, 1);


    // Difference is a matrix sized by t_width * t_width that has distance between template and image and save it in the cell
    finding_distance_kernel<<<dim_blocks_per_grid, dim_thread_per_block>>>(image, temp, t_width, difference, quantum_y);

    // Wait till result gets calculated
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("");

    t_width /= 2;
    blocks_per_grid_x = len / (thread_per_block * 2);

    while(level != len) {
        dim3 dim_blocks_per_grid(blocks_per_grid_x, 1);
        dim3 dim_thread_per_block(thread_per_block, 1, 1);
        reduction_kernel<<<dim_blocks_per_grid, dim_thread_per_block>>>(difference, level, t_width);
        cudaThreadSynchronize();
        CUT_CHECK_ERROR("");
        
        level *= 2;
        blocks_per_grid_x /= 2;
        // update level accordingly;
        if(blocks_per_grid_x == 0) {
            blocks_per_grid_x = 1;
            thread_per_block /= 2;
        }
    }
    float result;
    cudaMemcpy(&result, difference, sizeof(float), cudaMemcpyDeviceToHost); // First value in difference will have value
    return result;

}

float translated(float *image, int i_width, int i_height, float *temp, int t_width){
    

    float min_dist = FLT_MAX;
    unsigned int diff1 = i_height - t_width;
    unsigned int diff2 = i_width - t_width;
    size_t size_template = sizeof(float)*t_width*t_width;
    unsigned int A1 = i_height * diff2;
    unsigned int A2 = (i_width - diff2) * diff1;
    size_t size_distances = sizeof(float) * ((A1 + A2)) * 8 ;

    // Matix which stores differences between two cells
    float *d_difference;
    // Template 1: normal template
    // Nothing

    // Template 2: 90 Rotated template
    float *normal_90_rotated;

    // Template 3: 180 Rotated template
    float *normal_180_rotated;

    // Template 4: 270 Rotated template
    float *normal_270_rotated;

    // Template 5: filpped Template
    float *flipped_template;

    // Template 6: 90 rotated Template
    float *flipped_90_rotated;

    // Template 7: 180 Rotated template
    float *flipped_180_rotated;

    // Template 8: 270 Rotated Template 
    float *flipped_270_rotated;

    // Distances CPU accessible
    float *distances = (float*)(malloc(size_distances));

    unsigned int thread_block;

    if (t_width < 512)
    { 
	thread_block  = t_width;
    }
    else 
    {
	thread_block  = 512;
    }
    // Allocate in GPU memory 
    CUDA_SAFE_CALL(cudaMalloc(&normal_90_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&normal_180_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&normal_270_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&flipped_template, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&flipped_90_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&flipped_180_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&flipped_270_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&d_difference, size_template));
    
    // Make Rotation
    rotate_ccw_90(normal_90_rotated, temp, t_width, thread_block);
    rotate_ccw_90(normal_180_rotated, normal_90_rotated, t_width, thread_block);
    rotate_ccw_90(normal_270_rotated, normal_180_rotated, t_width, thread_block);
    flip_horizontal(flipped_template, temp, t_width, thread_block);
    rotate_ccw_90(flipped_90_rotated, flipped_template, t_width, thread_block);
    rotate_ccw_90(flipped_180_rotated, flipped_90_rotated, t_width, thread_block);
    rotate_ccw_90(flipped_270_rotated, flipped_180_rotated, t_width, thread_block);
    translated_y = i_width;

    int i = 0;
    for(int y = 0; y <= diff1; y++)
    {
        for(int x = 0; x <= diff2; x++)
        {
            // Original Image
            distances[i++] = finding_distance(image + x + y * i_width, temp, t_width, d_difference);
            distances[i++] = finding_distance(image + x + y * i_width, normal_90_rotated, t_width, d_difference);
            distances[i++] = finding_distance(image + x + y * i_width, normal_180_rotated, t_width, d_difference);
            distances[i++] = finding_distance(image + x + y * i_width, normal_270_rotated, t_width, d_difference);

            // Filped Image
            distances[i++] = finding_distance(image + x + y * i_width, flipped_template, t_width, d_difference);
            distances[i++] = finding_distance(image + x + y * i_width, flipped_90_rotated, t_width, d_difference);
            distances[i++] = finding_distance(image + x + y * i_width, flipped_180_rotated, t_width, d_difference);
            distances[i++] = finding_distance(image + x + y * i_width, flipped_270_rotated, t_width, d_difference);
            //printf("I COUNT %d \n", i);
        }
    }
    
    // CPU Calculation to finding minimum distance
    for(i--; i > 0 ; i--) 
    {
        //printf("I - %d - RESULT : %f - %d \n",i, distances[i], distances[i]);
        if(min_dist > distances[i]) 
        {
            min_dist = distances[i];
        }
        
    }
    
    //Free All
    CUDA_SAFE_CALL(cudaFree(normal_90_rotated));
    CUDA_SAFE_CALL(cudaFree(normal_180_rotated));
    CUDA_SAFE_CALL(cudaFree(normal_270_rotated));
    CUDA_SAFE_CALL(cudaFree(flipped_template));
    CUDA_SAFE_CALL(cudaFree(flipped_90_rotated));
    CUDA_SAFE_CALL(cudaFree(flipped_180_rotated));
    CUDA_SAFE_CALL(cudaFree(flipped_270_rotated));
    CUDA_SAFE_CALL(cudaFree(d_difference));
    free(distances);

    translated_y = 0;
    return min_dist;    
}

/* Returns the squared Euclidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with side length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */
float calc_min_dist(float *image, int i_width, int i_height, float *temp, int t_width) {
    // float* image and float* temp are pointers to GPU addressible memory
    // You MAY NOT copy this data back to CPU addressible memory and you MAY 
    // NOT perform any computation using values from image or temp on the CPU.
    // The only computation you may perform on the CPU directly derived from distance
    // values is selecting the minimum distance value given a calculated distance and a 
    // "min so far"
    if (i_width < t_width || i_height < t_width){ //impossible to match 
        printf("Invalid; size returning the maximum distance! \n");
        return UINT_MAX; // returning nonmatch immidetely
    }
    else if(image == NULL || temp == NULL){
        printf("Image passed the null object, exit with error");
        exit(0);
    }
    //cudaDeviceReset();
    if(i_width > t_width || i_height > t_width) {
        return translated(image, i_width, i_height, temp, t_width);
    }
    //cudaDeviceReset();
    float min_dist = FLT_MAX;
    size_t size_template = sizeof(float) * t_width * t_width;
    size_t size_distances = sizeof(float) * 8;
    
    // Matix which stores differences between two cells
    float *d_difference;

    // Template 1: normal template
    // Nothing

    // Template 2: 90 Rotated template
    float *normal_90_rotated;

    // Template 3: 180 Rotated template
    float *normal_180_rotated;

    // Template 4: 270 Rotated template
    float *normal_270_rotated;

    // Template 5: filpped Template
    float *flipped_template;

    // Template 6: 90 rotated Template
    float *flipped_90_rotated;

    // Template 7: 180 Rotated template
    float *flipped_180_rotated;

    // Template 8: 270 Rotated Template 
    float *flipped_270_rotated;
    
    // Distances CPU accessible
    float *distance = (float*)(malloc(size_distances));

    // Allocate in GPU memory 
    CUDA_SAFE_CALL(cudaMalloc(&normal_90_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&normal_180_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&normal_270_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&flipped_template, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&flipped_90_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&flipped_180_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&flipped_270_rotated, size_template));
    CUDA_SAFE_CALL(cudaMalloc(&d_difference, size_template));
     
    unsigned int thread_per_block;

    if (t_width < 512)
    { 
	thread_per_block = t_width;
    }
    else 
    {
	thread_per_block = 512;
    }

    // Make Rotation
    rotate_ccw_90(normal_90_rotated, temp, t_width, thread_per_block);
    rotate_ccw_90(normal_180_rotated, normal_90_rotated, t_width, thread_per_block);
    rotate_ccw_90(normal_270_rotated, normal_180_rotated, t_width, thread_per_block);
    flip_horizontal(flipped_template, temp, t_width, thread_per_block);
    rotate_ccw_90(flipped_90_rotated, flipped_template, t_width, thread_per_block);
    rotate_ccw_90(flipped_180_rotated, flipped_90_rotated, t_width, thread_per_block);
    rotate_ccw_90(flipped_270_rotated, flipped_180_rotated, t_width, thread_per_block);
    translated_y = i_width;

    
    distance[0] = finding_distance(image, temp, t_width, d_difference); // template
    distance[1] = finding_distance(image, normal_90_rotated, t_width, d_difference); // problem
    distance[2] = finding_distance(image, normal_180_rotated, t_width, d_difference); // template
    distance[3] = finding_distance(image, normal_270_rotated, t_width, d_difference);

    // Filped Image
    distance[4] = finding_distance(image, flipped_template, t_width, d_difference);
    distance[5] = finding_distance(image, flipped_90_rotated, t_width, d_difference);
    distance[6] = finding_distance(image, flipped_180_rotated, t_width, d_difference);
    distance[7] = finding_distance(image, flipped_270_rotated, t_width, d_difference);
    
   
    // Ready to do CPU calculation
    for(int i = 0; i < 8; i++) {
        if(min_dist > distance[i]) {
            min_dist = distance[i];
        }
    }

    //Free All
    CUDA_SAFE_CALL(cudaFree(normal_90_rotated));
    CUDA_SAFE_CALL(cudaFree(normal_180_rotated));
    CUDA_SAFE_CALL(cudaFree(normal_270_rotated));
    CUDA_SAFE_CALL(cudaFree(flipped_template));
    CUDA_SAFE_CALL(cudaFree(flipped_90_rotated));
    CUDA_SAFE_CALL(cudaFree(flipped_180_rotated));
    CUDA_SAFE_CALL(cudaFree(flipped_270_rotated));
    CUDA_SAFE_CALL(cudaFree(d_difference));
    free(distance);

    return min_dist;
}
