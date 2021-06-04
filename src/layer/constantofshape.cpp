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

#include "constantofshape.h"

namespace ncnn {

ConstantOfShape::ConstantOfShape()
{
    one_blob_only = true;
    support_inplace = false;
}

int ConstantOfShape::load_param(const ParamDict& pd)
{
    val = pd.get(0, 0.f);
    return 0;
}

int ConstantOfShape::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.w;
    if (dims == 1)
    {
        const int* shape_ptr = bottom_blob;
        int w = shape_ptr[0];
        size_t elemsize = sizeof(val);
        top_blob.create(w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        float* outptr = top_blob;
        for (int i = 0; i < w; i++)
        {
            outptr[i] = val;
        }
        return 0;
    }
    if (dims == 2)
    {
        const int* shape_ptr = bottom_blob;
        int h = shape_ptr[0];
        int w = shape_ptr[1];
        size_t elemsize = sizeof(val);
        top_blob.create(w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        float* outptr = top_blob;
        for (int i = 0; i < w * h; i++)
        {
            outptr[i] = val;
        }
        return 0;
    }
    if (dims == 3)
    {
        const int* shape_ptr = bottom_blob;
        int channels = shape_ptr[0];
        int h = shape_ptr[1];
        int w = shape_ptr[2];
        size_t elemsize = sizeof(val);
        top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);
            for (int i = 0; i < w * h; i++)
            {
                outptr[i] = val;
            }
        }
        return 0;
    }
    
    
}

} // namespace ncnn
