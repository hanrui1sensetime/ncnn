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

#include "range.h"

#include <math.h>

namespace ncnn {

Range::Range()
{
    one_blob_only = false;
    support_inplace = false;
}

int Range::forward(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Option& opt) const
{
    Mat start_blob = bottom_blobs[0];
    Mat limit_blob = bottom_blobs[1];
    Mat delta_blob = bottom_blobs[2];
    
    float start = start_blob[0];
    float limit = start_blob[1];
    float delta = start_blob[2];
    int size = (int) ((limit - start) / delta) + 1;
    size_t elemsize = bottom_blobs[0].elemsize;
    float* outptr = top_blob;
    top_blob.create(size, elemsize, opt.blob_allocator);
    float temp = start;
    while (temp <= limit)
    {
        outptr[0] = temp;
        outptr += 1;
        temp += delta;
    }

    return 0;
}

} // namespace ncnn
