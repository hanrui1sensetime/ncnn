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

#include "topk.h"

#include <functional>

namespace ncnn {
DEFINE_LAYER_CREATOR(TopK)

TopK::TopK()
{
    one_blob_only=false;
}

int TopK::forward(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Option& opt) const
{
    
    int size = bottom_blobs[0].total();
    int topk = bottom_blobs[1][0];
    top_blob.create(topk, 2, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* ptr = bottom_blobs[0];

    // partial sort topk with index
    // optional value
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(ptr[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    float* outptr = top_blob;
    float* valptr = outptr + topk;
    for (int i = 0; i < topk; i++)
    {
        outptr[i] = vec[i].first;
        valptr[i] = vec[i].second;
    }
    
    return 0;
}

} // namespace ncnn
