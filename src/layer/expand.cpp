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

#include "expand.h"

namespace ncnn {

Expand::Expand()
{
    one_blob_only = false;
    support_inplace = false;
}


int Expand::forward(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];

    
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int dims = bottom_blob.dims;

    bool _expand_w = false;
    bool _expand_h = false;
    bool _expand_c = false;

    const Mat& _axes = bottom_blobs[1];
    if (_axes.w > 0)
    {
        const int* axes_ptr = _axes;
        for (int i = 0; i < _axes.w; i++)
        {
            int axis = axes_ptr[i];
            if (axis < 0)
            {
                axis = dims + 1 + axis;
            }
            if (dims == 1 && axis == 1)
            {
                _expand_h = true;
            }
            if (dims == 1 && axis == 2)
            {
                _expand_w = true;
            }
            if (dims == 2 && axis == 1)
            {
                _expand_c = true;
            }
            if (dims == 2 && axis == 2)
            {
                _expand_h = true;
            }
            if (dims == 2 && axis == 3)
            {
                _expand_w = true;
            }
        }
    }

    top_blob = bottom_blob;

    if (dims == 1)
    {
        if (_expand_w && _expand_h)
        {
            top_blob = bottom_blob.reshape(1, w, 1, opt.blob_allocator);
        }
        else if (_expand_w)
        {
            top_blob = bottom_blob.reshape(1, w, opt.blob_allocator);
        }
        else if (_expand_h)
        {
            top_blob = bottom_blob.reshape(w, 1, opt.blob_allocator);
        }
    }

    if (dims == 2)
    {
        if (_expand_w)
        {
            top_blob = bottom_blob.reshape(1, w, h, opt.blob_allocator);
        }
        else if (_expand_h)
        {
            top_blob = bottom_blob.reshape(w, 1, h, opt.blob_allocator);
        }
        else if (_expand_c)
        {
            top_blob = bottom_blob.reshape(w, h, 1, opt.blob_allocator);
        }
    }

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
