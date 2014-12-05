/* CS61C Sp14 Project 3 Part 1: 
  *
  * You MUST implement the calc_min_dist() function in this file.
  *
  * You do not need to implement/use the swap(), flip_horizontal(), transpose(), or rotate_ccw_90()
  * functions, but you may find them useful. Feel free to define additional helper functions.
  * You do not need to implement/use the swap(), flip_horizontal(), transpose(),
  * or rotate_ccw_90() functions, but you may find them useful. Feel free to
  * define additional helper functions.
  */
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <emmintrin.h>
#include  <nmmintrin.h>
#include "digit_rec.h"
#include "utils.h"
#include "limits.h"

#define MAX_ROTATION 3 // 90, 180, 270

void flip_horizontal(float *arr, int width);
void swap(float *x, float *y);
void copy_matrix(float *dest, float *src, int len);
static unsigned int translated_y = 0;

/* Swaps the values pointed to by the pointers X and Y. */
void swap(float *x, float *y) {
    float temp = *x;
    *x = *y;
    *y = temp;
}

/* Flips the elements of a square array ARR across the y-axis. */
void flip_horizontal(float *arr, int width) {
    //return;
    __m128 top;
    __m128 bottom;
    int x;
    for(x = 0; (width - x) >= 4; x += 4) {
        for(int y = 0; y < width / 2 ; y += 1) {
            top = _mm_loadu_ps(&arr[y*width + x]);
            bottom = _mm_loadu_ps(&arr[width*(width - 1 - y) + x]);
            _mm_storeu_ps(&arr[y*width + x], bottom);
            _mm_storeu_ps(&arr[width*(width - 1 - y) + x], top);
        }
    }
    // Special case handling when the arry is not factor of 4
    if((width - x) > 0)
    {
        for(; x < width; x++) {
            for(int y = 0; y < width / 2 ; y += 1) {
                swap(&arr[width*(width - 1 - y) + x], &arr[y*width + x]);
            }
        }
    }
}

/* Returns the squared Euclidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with side length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */
void finding_distance(float *image, float *template, int t_width, float* distance) {
 //   return FLT_MAX;
    *distance = 0;
    __m128 distance_vector = _mm_setzero_ps( );
    float sum_floats1[4];
    int row_t1;
    int row_t2;
    int row_t3;
    int row_t4;

    int row_i1;
    int row_i2;
    int row_i3;
    int row_i4;
    unsigned int x;
    unsigned int y;
    unsigned int quantum_y = translated_y ? translated_y : t_width;
    __m128 sum1;
    __m128 sum2;
    __m128 sum3;
    __m128 sum4;

    __m128 row_image1;
    __m128 row_image2;
    __m128 row_image3;
    __m128 row_image4;

    __m128 row_template1;
    __m128 row_template2;
    __m128 row_template3;
    __m128 row_template4;

    for(y = 0; (t_width - y) >= 4; y+=4){ // c
        for(x = 0; (t_width - x) >= 4; x+=4){
            row_t1 = y*t_width + x; // changed
            row_t2 = row_t1 + t_width; // + t_width
            row_t3 = row_t2 + t_width;
            row_t4 = row_t3 + t_width;
            row_i1 = y*quantum_y + x;
            row_i2 = row_i1 + quantum_y;
            row_i3 = row_i2 + quantum_y;
            row_i4 = row_i3 + quantum_y;
            row_image1 = _mm_loadu_ps(image + row_i1);
            row_image2 = _mm_loadu_ps(image + row_i2);
            row_image3 = _mm_loadu_ps(image + row_i3);
            row_image4 = _mm_loadu_ps(image + row_i4);

            row_template1 = _mm_loadu_ps(template + row_t1); //CHANGE
            row_template2 = _mm_loadu_ps(template + row_t2);
            row_template3 = _mm_loadu_ps(template + row_t3);
            row_template4 = _mm_loadu_ps(template + row_t4);

            sum1 = _mm_sub_ps(row_template1, row_image1);
            sum2 = _mm_sub_ps(row_template2, row_image2);
            sum3 = _mm_sub_ps(row_template3, row_image3);
            sum4 = _mm_sub_ps(row_template4, row_image4);

            sum1 = _mm_mul_ps(sum1, sum1);
            sum2 = _mm_mul_ps(sum2, sum2);
            sum3 = _mm_mul_ps(sum3, sum3);
            sum4 = _mm_mul_ps(sum4, sum4);

            sum1 = _mm_add_ps(sum1, sum2);
            sum2 = _mm_add_ps(sum3, sum4);
            sum1 = _mm_add_ps(sum1, sum2);
            distance_vector = _mm_add_ps(distance_vector, sum1);
        }

        // Special case handling
        if ((t_width - x) > 0)
        {
            for(; x < t_width; x++){
                row_t1 = y*t_width + x; // changed
                row_t2 = row_t1 + t_width; // + t_width
                row_t3 = row_t2 + t_width;
                row_t4 = row_t3 + t_width;
                row_i1 = y*quantum_y + x;
                row_i2 = row_i1 + quantum_y;
                row_i3 = row_i2 + quantum_y;
                row_i4 = row_i3 + quantum_y;
                row_image1 = _mm_set_ps(image[row_i1], image[row_i2], image[row_i3], image[row_i4]);
                row_template1 = _mm_set_ps(template[row_t1], template[row_t2], template[row_t3], template[row_t4]);
                sum1 = _mm_sub_ps(row_template1, row_image1);
                sum1 = _mm_mul_ps(sum1, sum1);
                distance_vector = _mm_add_ps(distance_vector, sum1);
            }
        }

    }

    // edge case
    if ((t_width - y) > 0)
    {
        for(; y < t_width; y++){
            for(x = 0; (t_width - x) >= 4; x+=4){
                row_t1 = y*t_width + x; // changed
                row_i1 = y*quantum_y + x;
                row_image1 = _mm_loadu_ps(image + row_i1);
                row_template1 = _mm_loadu_ps(template + row_t1); //CHANGE
                sum1 = _mm_sub_ps(row_template1, row_image1);
                sum1 = _mm_mul_ps(sum1, sum1);
                distance_vector = _mm_add_ps(distance_vector, sum1);
            }
            // Special case handling
            if ((t_width - x) > 0)
            {
                for(; x < t_width; x++){
                    sum_floats1[0] = image[y*quantum_y + x];
                    sum_floats1[1] = template[y*t_width + x]; //CHANGE
                    *distance += (sum_floats1[0]  - sum_floats1[1] ) * (sum_floats1[0]   - sum_floats1[1] );
                }
            }
        }
    }

    _mm_storeu_ps(sum_floats1, distance_vector);
    *distance += sum_floats1[0] + sum_floats1[1] + sum_floats1[2] + sum_floats1[3];
}

// Caclculating without actually creating the matrix
void distance_ccw_90(float *image, float *template, int t_width, float* distance) {
 //   return FLT_MAX;
    *distance = 0;
    __m128 distance_vector = _mm_setzero_ps( );
    __m128 sum1;
    __m128 sum2;
    __m128 sum3;
    __m128 sum4;
    __m128 row_image1;
    __m128 row_image2;
    __m128 row_image3;
    __m128 row_image4;
    __m128 row_template1;
    __m128 row_template2;
    __m128 row_template3;
    __m128 row_template4;
    float sum_floats1[4];
    int row_t1;
    int row_t2;
    int row_t3;
    int row_t4;
    int row_i1;
    int row_i2;
    int row_i3;
    int row_i4;
    unsigned int x;
    unsigned int y;
    unsigned int quantum_y = translated_y ? translated_y : t_width;
    
    for(y = 0; (t_width - y) >= 4; y+=4) {
        for(x = 0; (t_width - x) >= 4; x+=4){
            row_t1 = (x + 1)*t_width - 1 - y; // changed
            row_t2 = row_t1 - 1; // + t_width
            row_t3 = row_t2 - 1;
            row_t4 = row_t3 - 1;
            row_i1 = y*quantum_y + x;
            row_i2 = row_i1 + quantum_y;
            row_i3 = row_i2 + quantum_y;
            row_i4 = row_i3 + quantum_y;
            row_image1 = _mm_loadu_ps(image + row_i1);
            row_image2 = _mm_loadu_ps(image + row_i2);
            row_image3 = _mm_loadu_ps(image + row_i3);
            row_image4 = _mm_loadu_ps(image + row_i4);

            row_template1 = _mm_set_ps(template[row_t1 + 3*t_width], template[row_t1 + 2*t_width], template[row_t1 + t_width], template[row_t1]); //CHANGE
            row_template2 = _mm_set_ps(template[row_t2 + 3*t_width], template[row_t2 + 2*t_width], template[row_t2 + t_width], template[row_t2]); //CHANGE
            row_template3 = _mm_set_ps(template[row_t3 + 3*t_width], template[row_t3 + 2*t_width], template[row_t3 + t_width], template[row_t3]); //CHANGE
            row_template4 = _mm_set_ps(template[row_t4 + 3*t_width], template[row_t4 + 2*t_width], template[row_t4 + t_width], template[row_t4]); //CHANGE

            sum1 = _mm_sub_ps(row_template1, row_image1);
            sum2 = _mm_sub_ps(row_template2, row_image2);
            sum3 = _mm_sub_ps(row_template3, row_image3);
            sum4 = _mm_sub_ps(row_template4, row_image4);

            sum1 = _mm_mul_ps(sum1, sum1);
            sum2 = _mm_mul_ps(sum2, sum2);
            sum3 = _mm_mul_ps(sum3, sum3);
            sum4 = _mm_mul_ps(sum4, sum4);

            sum1 = _mm_add_ps(sum1, sum2);
            sum2 = _mm_add_ps(sum3, sum4);
            sum1 = _mm_add_ps(sum1, sum2);
            distance_vector = _mm_add_ps(distance_vector, sum1);
        }
        // Special case handling
        if ((t_width - x) > 0)
        {
            for(; x < t_width; x++){
                row_t1 = (x+ 1)*t_width - 1 - y; // changed
                row_t2 = row_t1 - 1; // + t_width
                row_t3 = row_t2 - 1;
                row_t4 = row_t3 - 1;

                row_i1 = y*quantum_y + x;
                row_i2 = row_i1 + quantum_y;
                row_i3 = row_i2 + quantum_y;
                row_i4 = row_i3 + quantum_y;

                row_image1 = _mm_set_ps(image[row_i1], image[row_i2], image[row_i3], image[row_i4]);
                row_template1 = _mm_set_ps(template[row_t1], template[row_t2], template[row_t3], template[row_t4]);
                sum1 = _mm_sub_ps(row_template1, row_image1);
                sum1 = _mm_mul_ps(sum1, sum1);
                distance_vector = _mm_add_ps(distance_vector, sum1);
            }
        }
    }

    // edge case
    if ((t_width - y) > 0)
    {
        for(; y < t_width; y++){
            for(x = 0; (t_width - x) >= 4; x +=4){
                row_t1 = (x + 1)*t_width - 1 - y; // changed
                row_i1 = y*quantum_y + x;
                row_image1 = _mm_loadu_ps(image + row_i1);
                row_template1 = _mm_set_ps(template[row_t1 + 3*t_width], template[row_t1 + 2*t_width], template[row_t1 + t_width], template[row_t1]); //CHANGE
                sum1 = _mm_sub_ps(row_template1, row_image1);
                sum1 = _mm_mul_ps(sum1, sum1);
                distance_vector = _mm_add_ps(distance_vector, sum1);
            }
            // Special case handling
            if ((t_width - x) > 0)
            {
                for(; x < t_width; x++){
                    sum_floats1[0] = image[y*quantum_y + x];
                    sum_floats1[1] = template[(x + 1)*t_width - 1 - y]; //CHANGE
                    *distance += (sum_floats1[0] - sum_floats1[1]) * (sum_floats1[0]  - sum_floats1[1]);
                }
            }
        }
    }
    _mm_storeu_ps(sum_floats1, distance_vector);
    *distance += sum_floats1[0] + sum_floats1[1] + sum_floats1[2] + sum_floats1[3];
}

void distance_ccw_180(float *image, float *template, int t_width, float *distance) {
 //   return FLT_MAX;
    *distance = 0;
    float sum_floats1[4];
    int row_t1;
    int row_t2;
    int row_t3;
    int row_t4;
    int row_i1;
    int row_i2;
    int row_i3;
    int row_i4;
    unsigned int quantum_y = translated_y ? translated_y : t_width;

    __m128 distance_vector = _mm_setzero_ps( );
    __m128 sum1;
    __m128 sum2;
    __m128 sum3;
    __m128 sum4;

    __m128 row_image1;
    __m128 row_image2;
    __m128 row_image3;
    __m128 row_image4;

    __m128 row_template1;
    __m128 row_template2;
    __m128 row_template3;
    __m128 row_template4;


    unsigned int x;
    unsigned int y;
    for(y = 0; (t_width - y) >= 4; y+=4){
        for(x = 0; (t_width - x) >= 4; x+=4){
            row_t1 = (t_width - y)*t_width - 1 - x; // changed
            row_t2 = row_t1 - t_width;
            row_t3 = row_t2 - t_width;
            row_t4 = row_t3 - t_width;
            row_i1 = y*quantum_y + x;
            row_i2 = row_i1 + quantum_y;
            row_i3 = row_i2 + quantum_y;
            row_i4 = row_i3 + quantum_y;
            row_image1 = _mm_loadu_ps(image + row_i1);
            row_image2 = _mm_loadu_ps(image + row_i2);
            row_image3 = _mm_loadu_ps(image + row_i3);
            row_image4 = _mm_loadu_ps(image + row_i4);

            row_template1 = _mm_set_ps(template[row_t1 - 3], template[row_t1 - 2], template[row_t1 - 1], template[row_t1]); //CHANGE
            row_template2 = _mm_set_ps(template[row_t2 - 3], template[row_t2 - 2], template[row_t2 - 1], template[row_t2]); //CHANGE
            row_template3 = _mm_set_ps(template[row_t3 - 3], template[row_t3 - 2], template[row_t3 - 1], template[row_t3]); //CHANGE
            row_template4 = _mm_set_ps(template[row_t4 - 3], template[row_t4 - 2], template[row_t4 - 1], template[row_t4]); //CHANGE

            sum1 = _mm_sub_ps(row_template1, row_image1);
            sum2 = _mm_sub_ps(row_template2, row_image2);
            sum3 = _mm_sub_ps(row_template3, row_image3);
            sum4 = _mm_sub_ps(row_template4, row_image4);

            sum1 = _mm_mul_ps(sum1, sum1);
            sum2 = _mm_mul_ps(sum2, sum2);
            sum3 = _mm_mul_ps(sum3, sum3);
            sum4 = _mm_mul_ps(sum4, sum4);

            sum1 = _mm_add_ps(sum1, sum2);
            sum2 = _mm_add_ps(sum3, sum4);
            sum1 = _mm_add_ps(sum1, sum2);
            distance_vector = _mm_add_ps(distance_vector, sum1);
        }
        // Special case handling
        if ((t_width - x) > 0)
        {
            for(; x < t_width; x++){
                row_t1 = (t_width - y)*t_width - 1 - x; // changed
                row_t2 = row_t1 - t_width;
                row_t3 = row_t2 - t_width;
                row_t4 = row_t3 - t_width;
                row_i1 = y*quantum_y + x;
                row_i2 = row_i1 + quantum_y;
                row_i3 = row_i2 + quantum_y;
                row_i4 = row_i3 + quantum_y;

                row_image1 = _mm_set_ps(image[row_i1], image[row_i2], image[row_i3], image[row_i4]);
                row_template1  = _mm_set_ps(template[row_t1], template[row_t2], template[row_t3], template[row_t4]);
                sum1 = _mm_sub_ps(row_template1, row_image1);
                sum1 = _mm_mul_ps(sum1, sum1);
                distance_vector = _mm_add_ps(distance_vector, sum1);
            }
        }
        
    }

    // edge case
    if ((t_width - y) > 0)
    {
        for(; y < t_width; y++){

            for(x = 0; (t_width - x) >= 4; x+=4){
                row_t1 = (t_width - y)*t_width - 1 - x; // changed
                row_i1 = y*quantum_y + x;
                row_image1 = _mm_loadu_ps(image + row_i1);
                row_template1 = _mm_set_ps(template[row_t1 - 3], template[row_t1 - 2], template[row_t1 - 1], template[row_t1]); //CHANGE
                sum1 = _mm_sub_ps(row_template1, row_image1);
                sum1 = _mm_mul_ps(sum1, sum1);
                distance_vector = _mm_add_ps(distance_vector, sum1);
            }
            // Special case handling
            if ((t_width - x) > 0)
            {
                for(; x < t_width; x++){
                    sum_floats1[0] = image[y*quantum_y + x];
                    sum_floats1[1] = template[(t_width - y)*t_width - 1 - x]; //CHANGE
                    *distance += (sum_floats1[0] - sum_floats1[1]) * (sum_floats1[0]  - sum_floats1[1]);
                }
            }
        }
    }

    _mm_storeu_ps(sum_floats1, distance_vector);
    *distance += sum_floats1[0] + sum_floats1[1] + sum_floats1[2] + sum_floats1[3];
}

void distance_ccw_270(float *image, float *template, int t_width, float *distance) {
    //return FLT_MAX;
    *distance = 0;
    float sum_floats1[4];
    int row_t1;
    int row_t2;
    int row_t3;
    int row_t4;
    int row_i1;
    int row_i2;
    int row_i3;
    int row_i4;
    unsigned int x;
    unsigned int y;
    unsigned int quantum_y = translated_y ? translated_y : t_width;
    __m128 distance_vector = _mm_setzero_ps( );
    __m128 sum1;
    __m128 sum2;
    __m128 sum3;
    __m128 sum4;

    __m128 row_image1;
    __m128 row_image2;
    __m128 row_image3;
    __m128 row_image4;

    __m128 row_template1;
    __m128 row_template2;
    __m128 row_template3;
    __m128 row_template4;

    for(y = 0; (t_width - y) >= 4; y+=4){
        for(x = 0; (t_width - x) >= 4; x+=4){
            row_t1 = (t_width -(x+1))*t_width + y; // changed
            row_t2 = row_t1 + 1;
            row_t3 = row_t2 + 1;
            row_t4 = row_t3 + 1;
            row_i1 = y*quantum_y + x;
            row_i2 = row_i1 + quantum_y;
            row_i3 = row_i2 + quantum_y;
            row_i4 = row_i3 + quantum_y;
            row_image1 = _mm_loadu_ps(image + row_i1);
            row_image2 = _mm_loadu_ps(image + row_i2);
            row_image3 = _mm_loadu_ps(image + row_i3);
            row_image4 = _mm_loadu_ps(image + row_i4);

            row_template1 = _mm_set_ps(template[row_t1 - 3*t_width], template[row_t1 - 2*t_width], template[row_t1 - t_width], template[row_t1]); //CHANGE
            row_template2 = _mm_set_ps(template[row_t2 - 3*t_width], template[row_t2 - 2*t_width], template[row_t2 - t_width], template[row_t2]); //CHANGE
            row_template3 = _mm_set_ps(template[row_t3 - 3*t_width], template[row_t3 - 2*t_width], template[row_t3 - t_width], template[row_t3]); //CHANGE
            row_template4 = _mm_set_ps(template[row_t4 - 3*t_width], template[row_t4 - 2*t_width], template[row_t4 - t_width], template[row_t4]); //CHANGE
            
            sum1 = _mm_sub_ps(row_template1, row_image1);
            sum2 = _mm_sub_ps(row_template2, row_image2);
            sum3 = _mm_sub_ps(row_template3, row_image3);
            sum4 = _mm_sub_ps(row_template4, row_image4);

            sum1 = _mm_mul_ps(sum1, sum1);
            sum2 = _mm_mul_ps(sum2, sum2);
            sum3 = _mm_mul_ps(sum3, sum3);
            sum4 = _mm_mul_ps(sum4, sum4);

            sum1 = _mm_add_ps(sum1, sum2);
            sum2 = _mm_add_ps(sum3, sum4);
            sum1 = _mm_add_ps(sum1, sum2);
            distance_vector = _mm_add_ps(distance_vector, sum1);
        }
        
        // Special case handling
        if ((t_width - x) > 0)
        {
            for(; x < t_width; x++){
                row_t1 = (t_width -(x+1))*t_width + y; // changed
                row_t2 = row_t1 + 1;
                row_t3 = row_t2 + 1;
                row_t4 = row_t3 + 1;
                row_i1 = y*quantum_y + x;
                row_i2 = row_i1 + quantum_y;
                row_i3 = row_i2 + quantum_y;
                row_i4 = row_i3 + quantum_y;
                
                row_image1 = _mm_set_ps(image[row_i1], image[row_i2], image[row_i3], image[row_i4]);
                row_template1  = _mm_set_ps(template[row_t1], template[row_t2], template[row_t3], template[row_t4]);
                sum1 = _mm_sub_ps(row_template1, row_image1);
                sum1 = _mm_mul_ps(sum1, sum1);
                distance_vector = _mm_add_ps(distance_vector, sum1);
            }
        }
        
    }

    // edge case
    if ((t_width - y) > 0)
    {
        for(; y < t_width; y++){
            for(x = 0; (t_width - x) >= 4; x+=4){
                row_t1 = (t_width -(x+1))*t_width + y; // changed
                row_i1 = y*quantum_y + x;
                row_image1 = _mm_loadu_ps(image + row_i1);
                row_template1 = _mm_set_ps(template[row_t1 - 3*t_width], template[row_t1 - 2*t_width], template[row_t1 - t_width], template[row_t1]); //CHANGE
                sum1 = _mm_sub_ps(row_template1, row_image1);
                sum1 = _mm_mul_ps(sum1, sum1);
                sum1 = _mm_mul_ps(sum1, sum1);
                distance_vector = _mm_add_ps(distance_vector, sum1);
            }
            // Special case handling
            if ((t_width - x) > 0)
            {
                // Special case handling
                for(; x < t_width; x++){
                    sum_floats1[0] = image[y*quantum_y + x];
                    sum_floats1[1] = template[(t_width -(x+1))*t_width + y]; //CHANGE
                    *distance += (sum_floats1[0]  - sum_floats1[1] ) * (sum_floats1[0]   - sum_floats1[1] );
                }
            }
        }
    }

    _mm_storeu_ps(sum_floats1, distance_vector);
    *distance += sum_floats1[0] + sum_floats1[1] + sum_floats1[2] + sum_floats1[3];
}

float translated(float *image, int i_width, int i_height, float *template, int t_width){
    // Optimized version
    omp_set_num_threads(16);
    float min_dist = FLT_MAX;
    float distances[8]; 
    translated_y = i_width;
    int diff1 = i_height - t_width;
    int diff2 = i_width - t_width;
    float *copy_temp = (float*)malloc(sizeof(float)*t_width*t_width);
    copy_matrix(copy_temp, template, t_width * t_width);
    flip_horizontal(copy_temp, t_width);

    # pragma omp parallel for collapse(2) private(distances)
    for(int y = 0; y <= diff1; y++){
        for(int x = 0; x <= diff2; x++){
            #pragma omp parallel sections 
            {
                #pragma omp section 
                {
                    finding_distance(&image[x + y * i_width], template, t_width, &distances[0]);
                    #pragma omp critical
                    min_dist = (min_dist < distances[0]) ? min_dist : distances[0];
                }
                // Check rotation - flip horizontal - flip verticle
                // Unrolling applied
                // 90 degree calculation
                #pragma omp section 
                {
                    distance_ccw_90(&image[x + y * i_width], template, t_width, &distances[1]);
                    #pragma omp critical
                    min_dist = (min_dist < distances[1]) ? min_dist : distances[1];
                }
                #pragma omp section 
                {
                // 180 degree calculation
                    distance_ccw_180(&image[x + y * i_width], template, t_width, &distances[2]);
                    #pragma omp critical
                    min_dist = (min_dist < distances[2]) ? min_dist : distances[2];
                }
                #pragma omp section 
                {
                // 270 degree calculation
                    distance_ccw_270(&image[x + y * i_width], template, t_width, &distances[3]);
                    min_dist = (min_dist < distances[3]) ? min_dist : distances[3];
                }

                #pragma omp section 
                {
                    finding_distance(&image[x + y * i_width], copy_temp, t_width, &distances[4]);
                    #pragma omp critical
                    min_dist = (min_dist < distances[4]) ? min_dist : distances[4];
                }
                // Check rotation - flip horizontal - flip verticle
                // Unrolling applied
                // 90 degree calculation
                #pragma omp section 
                {
                    distance_ccw_90(&image[x + y * i_width], copy_temp, t_width, &distances[5]);
                    #pragma omp critical
                    min_dist = (min_dist < distances[5]) ? min_dist : distances[5];
                }
                #pragma omp section 
                {
                // 180 degree calculation
                    distance_ccw_180(&image[x + y * i_width], copy_temp, t_width, &distances[6]);
                    #pragma omp critical
                    min_dist = (min_dist < distances[6]) ? min_dist : distances[6];
                }
                #pragma omp section 
                {
                // 270 degree calculation
                    distance_ccw_270(&image[x + y * i_width], copy_temp, t_width, &distances[7]);
                    #pragma omp critical
                    min_dist = (min_dist < distances[7]) ? min_dist : distances[7];
                }
            }
        }
    }
    free(copy_temp);
	translated_y = 0;
	return min_dist;	
}
void copy_matrix(float *dest, float *src, int len) {
    //return;
    //#pragma omp parallel for
    __m128 src_vector;

    int i;
    for(i = 0; (i + 4) <= len; i+=4){
        src_vector = _mm_loadu_ps(src + i);
        _mm_storeu_ps(dest + i, src_vector);
    }

    // if template is not factor of 4
    if(i != len)
    {
        for(; i < len; i++){
            dest[i] = src[i];
        }
    }
}
float calc_min_dist(float *image, int i_width, int i_height, float *template, int t_width) {
    if(i_width > t_width || i_height > t_width) {
        return translated(image, i_width, i_height, template, t_width);
    }
    if (i_width < t_width || i_height < t_width){ //impossible to match 
        printf("Invalid; size returning the maximum distance! \n");
        return UINT_MAX; // returning nonmatch immidetely
    }
    else if(image == NULL || template == NULL){
        printf("Image passed the null object, exit with error");
        exit(0);
    }
    
    omp_set_num_threads(8);
    float min_dist = FLT_MAX;
    float distances[8]; 
    float *copy_temp = (float*)malloc(sizeof(float)*t_width*t_width);
    copy_matrix(copy_temp, template, t_width * t_width);
    flip_horizontal(copy_temp, t_width);
    #pragma omp parallel 
    {
        #pragma omp sections 
        {
            #pragma omp section 
            {
                finding_distance(image, template, t_width, &distances[0]);
                #pragma omp critical
                min_dist = (min_dist < distances[0]) ? min_dist : distances[0];
            }
            // Check rotation - flip horizontal - flip verticle
            // Unrolling applied
            // 90 degree calculation
            #pragma omp section 
            {
                distance_ccw_90(image, template, t_width, &distances[1]);
                #pragma omp critical
                min_dist = (min_dist < distances[1]) ? min_dist : distances[1];
            }
            #pragma omp section 
            {
            // 180 degree calculation
                distance_ccw_180(image, template, t_width, &distances[2]);
                #pragma omp critical
                min_dist = (min_dist < distances[2]) ? min_dist : distances[2];
            }
            #pragma omp section 
            {
            // 270 degree calculation
                distance_ccw_270(image, template, t_width, &distances[3]);
                #pragma omp critical
                min_dist = (min_dist < distances[3]) ? min_dist : distances[3];
            }

            #pragma omp section 
            {
                finding_distance(image, copy_temp, t_width, &distances[4]);
                #pragma omp critical
                min_dist = (min_dist < distances[4]) ? min_dist : distances[4];
            }
            // Check rotation - flip horizontal - flip verticle
            // Unrolling applied
            // 90 degree calculation
            #pragma omp section 
            {
                distance_ccw_90(image, copy_temp, t_width, &distances[5]);
                #pragma omp critical
                min_dist = (min_dist < distances[5]) ? min_dist : distances[5];
            }
            #pragma omp section 
            {
            // 180 degree calculation
                distance_ccw_180(image, copy_temp, t_width, &distances[6]);
                #pragma omp critical
                min_dist = (min_dist < distances[6]) ? min_dist : distances[6];
            }
            #pragma omp section 
            {
            // 270 degree calculation
                distance_ccw_270(image, copy_temp, t_width, &distances[7]);
                #pragma omp critical
                min_dist = (min_dist < distances[7]) ? min_dist : distances[7];
            }
        }
    }
    free(copy_temp);
    return min_dist;
}