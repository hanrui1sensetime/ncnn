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

#include "shape.h"

namespace ncnn {

Shape::Shape()
{
    one_blob_only = true;
    support_inplace = false;
}


int Shape::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int w = bottom_blob.w;
    size_t elemsize = sizeof(bottom_blob.w);
    top_blob.create(dims, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;
    int* outptr = top_blob;
    if (dims == 1)
    {
        outptr[0] = w;
        return 0;
    }
    if (dims == 2)
    {
        int h = bottom_blob.h;
        outptr[0] = h;
        outptr[1] = w;
        return 0;
    }
    if (dims == 3)
    {
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        outptr[0] = channels;
        outptr[1] = h;
        outptr[2] = w;
        return 0;
    }
}

} // namespace ncnn
