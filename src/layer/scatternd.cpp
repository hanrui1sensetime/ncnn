// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "scatternd.h"

namespace ncnn {

ScatterND::ScatterND()
{
    one_blob_only = false;
    support_inplace = false;
}


int ScatterND::forward(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& indice_blob = bottom_blobs[1];
    const Mat& update_blob = bottom_blobs[2];
    
    int dim_data = bottom_blob.dims;
    int dim_indice = indice_blob.dims;
    if (dim_data == 1)
    {
        size_t elemsize = bottom_blob.elemsize;
        int size = bottom_blob.w;
        top_blob.create(size, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned char* outptr = top_blob;
        const unsigned char* ptr = bottom_blob;
        memcpy(outptr, ptr, size * elemsize);

        if (dim_indice == 2)
        {
            if (indice_blob.w > 1)
            {
                printf("Wrong shape, indice[-1] should not greater than the data rank.");
                return -100;
            }
            if (indice_blob.w == 1)
            {
                float* outptr = top_blob;
                const float* ptr = update_blob;
                for (int i = 0; i < indice_blob.h; i++)
                {
                    const float* indptr = indice_blob.row(i);
                    outptr[(int) indptr[0]] = ptr[i];
                }
            }
            else
            {
                printf("wrong shape!");
                return -100;
            }
        }
        if (dim_indice == 3)
        {
            if (indice_blob.w > 1)
            {
                printf("Wrong shape, indice[-1] should not greater than the data rank.");
                return -100;
            }
            if (indice_blob.w == 1)
            {
                float* outptr = top_blob;
                const float* ptr = update_blob;
                int ind_c = indice_blob.c;
                int ind_h = indice_blob.h;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < ind_c; q++)
                {
                    const int* indptr = indice_blob.channel(q);
                    for (int i = 0; i < ind_h; i++)
                    {
                        outptr[indptr[i]] = ptr[q * ind_h + i];
                    }
                }
            }
            else
            {
                printf("wrong shape!");
                return -100;
            }
        }
    }
    if (dim_data == 2)
    {
        size_t elemsize = bottom_blob.elemsize;
        int bottom_w = bottom_blob.w;
        int bottom_h = bottom_blob.h;
        top_blob.create(bottom_w, bottom_h ,elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned char* outptr = top_blob;
        const unsigned char* ptr = bottom_blob;
        int size = bottom_blob.h * bottom_blob.w;
        memcpy(outptr, ptr, size * elemsize);

        if (dim_indice == 2)
        {
            if (indice_blob.w > 2)
            {
                printf("Wrong shape, indice[-1] should not greater than the data rank.");
                return -100;
            }
            if (indice_blob.w == 1)
            {
                for (int i = 0; i < indice_blob.h; i++)
                {
                    const float* indptr = indice_blob.row(i);
                    const float* ptr = update_blob.row(i);
                    float* outptr = top_blob.row((int) indptr[0]);
                    for (int j = 0; j < bottom_blob.w; j++)
                    {
                        outptr[j] = ptr[j]; 
                    }
                }
            }
            if (indice_blob.w == 2)
            {
                float* outptr = top_blob;
                const float* ptr = update_blob;
                for (int i = 0; i < indice_blob.h; i++)
                {
                    const float* indptr = indice_blob.row(i);
                    outptr[((int) indptr[0]) * bottom_w +((int) indptr[1])] = ptr[i];
                }
            }
            else
            {
                printf("wrong shape!");
                return -100;
            }
        }
        if (dim_indice == 3)
        {
            if (indice_blob.w > 2)
            {
                printf("Wrong shape, indice[-1] should not greater than the data rank.");
                return -100;
            }
            if (indice_blob.w == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < indice_blob.c; q++)
                {
                    for (int i = 0; i < indice_blob.h; i++)
                    {
                        const float* indptr = indice_blob.channel(q).row(i);
                        const float* ptr = update_blob.channel(q).row(i);
                        float* outptr = top_blob.row((int) indptr[0]);
                        for (int j = 0; j < bottom_blob.w; j++)
                        {
                            outptr[j] = ptr[j]; 
                        }
                    }
                }
            }
            if (indice_blob.w == 2)
            {
                float* outptr = top_blob;
                const float* ptr = update_blob;
                int ind_c = indice_blob.c;
                int ind_h = indice_blob.h;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < ind_c; q++)
                {
                    for (int i = 0; i < ind_h; i++)
                    {
                        const float* indptr = indice_blob.channel(q).row(i);
                        outptr[((int)indptr[0]) * bottom_w + ((int)indptr[1])] = ptr[q * ind_h + i];
                    }
                }
            }
            else
            {
                printf("wrong shape!");
                return -100;
            }
        }
    }
    if (dim_data == 3)
    {
        size_t elemsize = bottom_blob.elemsize;
        int bottom_w = bottom_blob.w;
        int bottom_h = bottom_blob.h;
        int bottom_c = bottom_blob.c;
        top_blob.create(bottom_w, bottom_h, bottom_c ,elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        int size = bottom_w * bottom_h * bottom_c;
        unsigned char* outptr = top_blob;
        const unsigned char* ptr = bottom_blob;
        memcpy(outptr, ptr, size * elemsize);

        if (dim_indice == 2)
        {
            if (indice_blob.w > 3)
            {
                printf("Wrong shape, indice[-1] should not greater than the data rank.");
                return -100;
            }
            if (indice_blob.w == 1)
            {
                for (int i = 0; i < indice_blob.h; i++)
                {
                    const float* indptr = indice_blob.row(i);
                    const float* ptr = update_blob.channel(i);
                    float* outptr = top_blob.channel((int) indptr[0]);
                    for (int j = 0; j < bottom_h * bottom_w; j++)
                    {
                        outptr[j] = ptr[j]; 
                    }
                }
            }
            if (indice_blob.w == 2)
            {
                for (int i = 0; i < indice_blob.h; i++)
                {
                    const float* indptr = indice_blob.row(i);
                    const float* ptr = update_blob.row(i);
                    float *outptr = top_blob.channel((int)indptr[0]).row((int)indptr[1]);
                    for (int j = 0; j < bottom_w; j++)
                    {
                        outptr[j] = ptr[j];
                    }
                }
            }
            if (indice_blob.w == 3)
            {
                for (int i = 0; i < indice_blob.h; i++)
                {
                    const float* indptr = indice_blob.row(i);
                    const float* ptr = update_blob.row(i);
                    float *outptr = top_blob.channel((int)indptr[0]);
                    int offset = (int) indptr[1] * bottom_w + (int) indptr[2];
                    outptr[offset] = ptr[0];
                }
            }
            else
            {
                printf("wrong shape!");
                return -100;
            }
        }
        if (dim_indice == 3)
        {
            if (indice_blob.w > 3)
            {
                printf("Wrong shape, indice[-1] should not greater than the data rank.");
                return -100;
            }
            if (indice_blob.w == 1)
            {
                printf("update_blob dim > 3 is not implemented now!");
                return -100;
            }
            if (indice_blob.w == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < indice_blob.c; q++)
                {
                    for (int i = 0; i < indice_blob.h; i++)
                    {
                        const float* indptr = indice_blob.channel(q).row(i);
                        const float* ptr = update_blob.channel(q).row(i);
                        float* outptr = top_blob.channel((int)indptr[0]).row((int)indptr[1]);
                        for (int j = 0; j < bottom_blob.w; j++)
                        {
                            outptr[j] = ptr[j]; 
                        }
                    }
                }
            }
            if (indice_blob.w == 3)
            {
                const float* ptr = update_blob;
                int ind_c = indice_blob.c;
                int ind_h = indice_blob.h;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < ind_c; q++)
                {
                    for (int i = 0; i < ind_h; i++)
                    {
                        const float* indptr = indice_blob.channel(q).row(i);
                        float* outptr = top_blob.channel((int)indptr[0]);
                        int offset = (int) indptr[1] * bottom_w + (int) indptr[2];
                        outptr[offset] = ptr[q * ind_h + i];
                    }
                }
            }
            else
            {
                printf("wrong shape!");
                return -100;
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
