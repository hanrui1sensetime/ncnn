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

#include "gather.h"

namespace ncnn {

Gather::Gather()
{
    one_blob_only = false;
    support_inplace = false;
}

int Gather::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

int Gather::forward(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& indices = bottom_blobs[1];
    int dims = bottom_blob.dims;
    int indices_dims = indices.dims;
    size_t elemsize = bottom_blob.elemsize;
    int positive_axis = axis < 0 ? dims + axis : axis;
    if (indices_dims > 1)
    {
        // To do: check error hint use printf?
        printf("The indice dims greater than 1 is not implemented!\n");
        // we want to return an error code for not implemented! (-100 need to check if it's right)
        return -100;
    }


    const int* indices_ptr = indices;

    if (dims == 1) // positive_axis == 0
    {
        // To do: check will the dims of the indices be zero?
        int w = indices.w;
        top_blob.create(w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        float* outptr = top_blob;
        for (int i = 0; i < w; i++)
        {
            int indice = indices_ptr[i];
            outptr[i]=ptr[indice];
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        top_blob.create(w, indices.w, elemsize, opt.blob_allocator);
        // To do: check the top_blob shape is right for onnx.gather?
        // Thinking that if it is right?
        // w -> w
        // h -> indices.w
        // h * w -> indices.w * w
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        float* outptr = top_blob;
        for (int i = 0; i < indices.w; i++)
        {
            for (int j = 0; j < w; j++)
            {
                int selected = indices_ptr[i];
                outptr[i * w + j] = ptr[selected * w + j];
            }
        } 

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        top_blob.create(h, indices.w, elemsize, opt.blob_allocator);
        // To do: check the top_blob shape is right for onnx.gather?
        // Thinking that if it is right?
        // w -> h
        // h -> indices.w
        // h * w -> indices.w * h
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        float* outptr = top_blob;
        for (int i = 0; i < indices.w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                int selected = indices_ptr[i];
                outptr[i * h + j] = ptr[j * w + selected];
            }
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        top_blob.create(w, h, indices.w, elemsize, opt.blob_allocator);
        // To do: check the order of the axis, refers to dims==2 cases.
        if (top_blob.empty())
            return -100;
    
        for (int i = 0; i < indices.w; i++)
        {
            int selected = indices_ptr[i];
            const unsigned char* ptr = bottom_blob.channel(selected);
            unsigned char* outptr = top_blob.channel(i);
            // To do: check using unsigned char to copy is right?
            // Or using assign one by one?
            memcpy(outptr, ptr, w * h * elemsize);

        }
        return 0;
    }

    if (dims == 3 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        top_blob.create(w, channels, indices.w, elemsize, opt.blob_allocator);
        // To do: check the order of the axis, refers to dims==2 cases.
        #pragma omp parallel for num_threads(opt.num_threads)
        // try to use parrllel programming
        // To do: check if the using is right.
        for (int i = 0; i < indices.w; i++)
        {
            int selected = indices_ptr[i];
            float* outptr = top_blob.channel(i);
            for (int j = 0; j < channels; j++)
            {
                const float* ptr = bottom_blob.channel(j);
                for (int k = 0; k < w; k++)
                {
                    outptr[j * w + k] = ptr[selected * w + k];
                }
            }
        }

        return 0;
    }

    if (dims == 3 && positive_axis == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        top_blob.create(h, channels, indices.w, elemsize, opt.blob_allocator);
        // To do: check the order of the axis, refers to dims==2 cases.
        #pragma omp parallel for num_threads(opt.num_threads)
        // try to use parrllel programming
        // To do: check if the using is right.
        for (int i = 0; i < indices.w; i++)
        {
            int selected = indices_ptr[i];
            float* outptr = top_blob.channel(i);
            for (int j = 0; j < channels; j++)
            {
                const float* ptr = bottom_blob.channel(j);
                for (int k = 0; k < h; k++)
                {
                    outptr[j * h + k] = ptr[k * w + selected];
                }
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
