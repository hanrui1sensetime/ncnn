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

#include "tileonnx.h"

namespace ncnn {

TileOnnx::TileOnnx()
{
    one_blob_only = false;
    support_inplace = false;
}

int TileOnnx::forward(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& tile_blob = bottom_blobs[1];
    
    int data_dim = tile_blob.w;
    if (bottom_blob.dims != data_dim)
    {
        printf("The shape of tile_blob should fit the input tensor dimensions.");
        return -100;
    }
    if (data_dim == 1)
    {
        int w = bottom_blob.w;
        size_t elemsize = bottom_blob.elemsize;
        const int* tile_ptr = tile_blob;
        int tiles_w = tile_ptr[0];

        top_blob.create(w * tiles_w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        float* outptr = top_blob;

        for (int p = 0; p < tiles_w; p++)
        {
            for (int i = 0; i < w; i++)
            {
                outptr[p * w + i] = ptr[i];
            }
        }
        return 0;
        
    }
    if (data_dim == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        const int* tile_ptr = tile_blob;
        int tiles_w = tile_ptr[1];
        int tiles_h = tile_ptr[0];

        top_blob.create(w * tiles_w, h * tiles_h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        
        const float* ptr = bottom_blob;
        float* outptr = top_blob;

        for (int q = 0; q < h * tiles_h; q++)
        {
            for (int p = 0; p < w * tiles_w; p++)
            {
                outptr[q*w*tiles_w + p] = ptr[(q % h) * w + (p % w)];
            }
        }
        return 0;
    }
    if (data_dim == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        size_t elemsize = bottom_blob.elemsize;

        const int* tile_ptr = tile_blob;
        int tiles_c = tile_ptr[0];
        int tiles_h = tile_ptr[1];
        int tiles_w = tile_ptr[2];

        top_blob.create(w * tiles_w, h * tiles_h, channels * tiles_c, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < tiles_c * channels; r++)
        {
            float* outptr = top_blob.channel(r);
            const float* ptr = bottom_blob.channel(r % channels);
            for (int q = 0; q < tiles_h * h; q++)
            {
                for (int p = 0; p < tiles_w * w; p++)
                {
                    outptr[q*w*tiles_w + p] = ptr[(q % h) * w + (p % w)];
                }
            }
        }
        return 0;
    }

}

} // namespace ncnn
