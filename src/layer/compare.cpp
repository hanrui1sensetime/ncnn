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

#include "compare.h"

#include <math.h>
#define EPS 1e-9
namespace ncnn {

Compare::Compare()
{
    one_blob_only = false;
    support_inplace = false;
}

int Compare::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    return 0;
}


// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

template<typename Op>
static int condition_compare(const Mat& left, const Mat& right, Mat& result, const Option& opt)
{
    
    Op op;

    int w = left.w;
    int h = left.h;
    int channels = left.c;
    int size = w * h;
    size_t elemsize = left.elemsize;

    int w1 = right.w;
    int h1 = right.h;
    int channels1 = right.c;
    int size1 = w1 * h1;
    Mat temp_result;    
    if (left.dims == 3)
    {
        if (left.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                
                // To do: check can opencv Mat be bool matrix?
                temp_result.create(w, h, channels, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool all_true = true;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    if (!all_true)
                    {
                        continue;
                    }
                    const float* leftptr = left.channel(q);
                    const float* rightptr = right.channel(q);
                    bool* tempptr = temp_result.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        if (!all_true)
                            continue;
                        tempptr[i] = op(leftptr[i], rightptr[0]);
                        if (!tempptr[i])
                        {
                            all_true = false;
                        }

                    }
                }
                return all_true;

            }
            //To do: need to change.
            if (w1 == w && h1 == h && channels1 == 1)
            {
                // special type 2
                Mat temp_result;
                temp_result.create(w, h, channels, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool all_true = true;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    if (!all_true)
                    {
                        continue;
                    }
                    const float* leftptr = left.channel(q);
                    const float* rightptr = right;
                    bool* tempptr = temp_result.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        if (!all_true)
                            continue;
                        tempptr[i] = op(leftptr[i], rightptr[i]);
                        if (!tempptr[i])
                        {
                            all_true = false;
                        }

                    }
                }
                return all_true;
               
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                Mat temp_result;
                temp_result.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;

                bool all_true = true;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    if (!all_true)
                        continue;
                    const float* leftptr = left.channel(q);
                    const float* rightptr = right.channel(q);
                    bool* tempptr = temp_result.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        if (!all_true)
                        {
                            continue;
                        }
                        tempptr[i] = op(leftptr[0], rightptr[i]);
                        if (!tempptr[i])
                        {
                            all_true = false;
                        }
                    }
                    
                }

                return all_true;
            }

            if (w1 == w && h1 == h && channels == 1)
            {
                // special type 4
                Mat temp_result;
                temp_result.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;

                bool all_true = true;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    if (!all_true)
                        continue;
                    const float* leftptr = left;
                    const float* rightptr = right.channel(q);
                    bool* tempptr = temp_result.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        if (!all_true)
                            continue;
                        tempptr[i] = op(leftptr[i], rightptr[i]);
                        if (!tempptr[i])
                        {
                            all_true = false;
                        }
                    }
                }

                return all_true;
            }

            if (w != 1 && w1 == 1 && h1 == h && channels1 == channels)
            {
                // special type 5
                Mat temp_result;
                temp_result.create(w, h, channels, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool  all_true = true;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    if (!all_true)
                        continue;
                    const float* leftptr = left.channel(q);
                    const float* rightptr = right.channel(q);
                    bool* tempptr = temp_result.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        if (!all_true)
                            continue;
                        const float b0 = rightptr[y];
                        for (int x = 0; x < w; x++)
                        {
                            if (!all_true)
                                continue;
                            tempptr[x] = op(leftptr[x], b0);
                            if (!tempptr[x])
                            {
                                all_true = false;
                            }
                        }

                        leftptr += w;
                        tempptr += w;
                    }

                }

                return all_true;
            }

            if (w1 == w && h != 1 && h1 == 1 && channels1 == channels)
            {
                // special type 6
                Mat temp_result;
                temp_result.create(w, h, channels, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool all_true = true;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    if (!all_true)
                        continue;
                    const float* leftptr = left.channel(q);
                    const float* rightptr = right.channel(q);
                    bool* tempptr = temp_result.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        if (!all_true)
                            continue;
                        for (int x = 0; x < w; x++)
                        {
                            if (!all_true)
                                continue;
                            tempptr[x] = op(leftptr[x], rightptr[x]);
                            if (!tempptr[x])
                            {
                                all_true = false;
                            }
                        }
                        leftptr += w;
                        rightptr += w;
                    }
                }

                return all_true;
            }

            if (w1 != 1 && w == 1 && h1 == h && channels1 == channels)
            {
                // special type 7
                Mat temp_result;
                temp_result.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool all_true = true;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    if (!all_true)
                        continue;
                    const float* leftptr = left.channel(q);
                    const float* rightptr = right.channel(q);
                    bool* tempptr = temp_result.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        if (!all_true)
                            continue;
                        const float a0 = leftptr[y];
                        for (int x = 0; x < w1; x++)
                        {
                            if (!all_true)
                                continue;
                            tempptr[x] = op(a0, rightptr[x]);
                            if (!tempptr[x])
                            {
                                all_true = false;
                            }
                        }

                        rightptr += w1;
                        tempptr += w1;
                    }

                }

                return all_true;
            }

            if (w1 == w && h1 != 1 && h == 1 && channels1 == channels)
            {
                // special type 8
                Mat temp_result;
                temp_result.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool all_true = true;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    if (!all_true)
                        continue;
                    const float* leftptr = left.channel(q);
                    const float* rightptr = right.channel(q);
                    bool* tempptr = temp_result.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        if (!all_true)
                            continue;
                        for (int x = 0; x < w1; x++)
                        {
                            if (!all_true)
                                continue;
                            tempptr[x] = op(leftptr[x], rightptr[x]);
                            if (!tempptr[x])
                            {
                                all_true = false;
                            }
                        }

                        rightptr += w1;
                        tempptr += w1;
                    }

                }

                return all_true;
            }

            // type 19

            temp_result.create(w, h, channels, elemsize, opt.blob_allocator);
            if (temp_result.empty())
                return -100;
            bool all_true = true;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                if (!all_true)
                    continue;
                const float* leftptr = left.channel(q);
                const float* rightptr = right.channel(q);
                bool* tempptr = temp_result.channel(q);

                for (int i = 0; i < size; i++)
                {
                    if (!all_true)
                        continue;
                    tempptr[i] = op(leftptr[i], rightptr[i]);
                    if (!tempptr[i])
                    {
                        all_true = false;
                    }
                }
            }

            return all_true;
        }

        temp_result.create(w, h, channels, elemsize, opt.blob_allocator);
        if (temp_result.empty())
            return -100;
        bool all_true = true;
        if (right.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                if (!all_true)
                    continue;
                const float* leftptr = left.channel(q);
                const float* rightptr = right.row(q);
                bool* tempptr = temp_result.channel(q);

                for (int y = 0; y < h; y++)
                {
                    if (!all_true)
                        continue;
                    const float b0 = rightptr[y];
                    for (int x = 0; x < w; x++)
                    {
                        if (!all_true)
                            continue;
                        tempptr[x] = op(leftptr[x], b0);
                        if (!tempptr[x])
                        {
                            all_true = false;
                        }
                    }
                    leftptr += w;
                    tempptr += w;
                }
            }

            return all_true;
        }

        if (right.dims == 1)
        {
            if (right.w == 1)
            {
                // type 16
                const float b0 = right[0];
                bool all_true = true;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    if (!all_true)
                        continue;
                    const float* leftptr = left.channel(q);
                    bool* tempptr = temp_result.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        if (!all_true)
                            continue;
                        tempptr[i] = op(leftptr[i], b0);
                        if (!tempptr[i])
                        {
                            all_true = false;
                        }
                    }
                }

                return all_true;
            }

            // type 17
            bool all_true = true;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                if (!all_true)
                    continue;
                const float* leftptr = left.channel(q);
                const float b0 = right[q];
                bool* tempptr = temp_result.channel(q);

                for (int i = 0; i < size; i++)
                {
                    if (!all_true)
                        continue;
                    tempptr[i] = op(leftptr[i], b0);
                    if (!tempptr[i])
                    {
                        all_true = false;
                    }
                }
            }

            return all_true;
        }
    }
    else if (left.dims == 2)
    {
        if (right.dims == 3)
        {
            // type 14
            temp_result.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (temp_result.empty())
                return -100;
            bool all_true = true;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                if (!all_true)
                    continue;
                const float* leftptr = left.row(q);
                const float* rightptr = right.channel(q);
                bool* tempptr = temp_result.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    if (!all_true)
                        continue;
                    const float a0 = leftptr[y];
                    for (int x = 0; x < w1; x++)
                    {
                        if (!all_true)
                            continue;
                        tempptr[x] = op(a0, rightptr[x]);
                        if (!tempptr[x])
                        {
                            all_true = false;
                        }
                    }
                    rightptr += w1;
                    tempptr += w1;
                }
            }

            return all_true;
        }


        if (right.dims == 2)
        {
            // type 13
            temp_result.create(w, h, elemsize, opt.blob_allocator);
            if (temp_result.empty())
                return -100;
            bool all_true = true;
            for (int i = 0; i < size; i++)
            {
                if (!all_true)
                    continue;
                temp_result[i] = op(left[i], right[i]);
                if (!temp_result[i])
                {
                    all_true = false;
                }
            }

            return all_true;
        }

        if (right.dims == 1)
        {
            if (right.w == 1)
            {
                // type 11
                temp_result.create(w, h, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool all_true = true;
                const float b0 = right[0];
                for (int i = 0; i < size; i++)
                {
                    if (!all_true)
                        continue;
                    temp_result[i] = op(left[i], b0);
                    if (!temp_result[i])
                    {
                        all_true = false;
                    }
                }

                return all_true;
            }

            // type 12
            const float* leftptr = left;
            bool* tempptr = temp_result;
            bool all_true = true;
            for (int y = 0; y < h; y++)
            {
                if (!all_true)
                    continue;
                const float b0 = right[y];
                for (int x = 0; x < w; x++)
                {
                    if (!all_true)
                        continue;
                    tempptr[x] = op(leftptr[x], b0);
                    if (!tempptr[x])
                    {
                        all_true = false;
                    }
                }
                leftptr += w;
                tempptr += w;
            }

            return all_true;
        }
    }
    else if (left.dims == 1)
    {
        if (left.w == 1)
        {
            if (right.dims == 3)
            {
                // type 4
                temp_result.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool all_true = true;
                const float a0 = left[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    if (!all_true)
                        continue;
                    const float* rightptr = right.channel(q);
                    bool* tempptr = temp_result.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        if (!all_true)
                            continue;
                        tempptr[i] = op(a0, rightptr[i]);
                        if (!tempptr[i])
                        {
                            all_true = false;
                        }
                    }
                }

                return all_true;
            }

            if (right.dims == 2)
            {
                // type 3
                temp_result.create(w1, h1, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool all_true = true;
                const float a0 = left[0];
                for (int i = 0; i < size1; i++)
                {
                    if (!all_true)
                        continue;
                    temp_result[i] = op(a0, right[i]);
                    if (!temp_result[i])
                    {
                        all_true = false;
                    }
                }

                return all_true;
            }

            if (right.dims == 1)
            {
                // type 2
                temp_result.create(w1, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool all_true = false;
                const float a0 = left[0];
                for (int i = 0; i < w1; i++)
                {
                    if (!all_true)
                        continue;
                    temp_result[i] = op(a0, right[i]);
                    if (!temp_result[i])
                    {
                        all_true = false;
                    }
                }

                return all_true;
            }
        }

        if (right.dims == 3)
        {
            // type 9
            temp_result.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (temp_result.empty())
                return -100;
            bool all_true = true;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                if (!all_true)
                    continue;
                const float a0 = left[q];
                const float* rightptr = right.channel(q);
                bool* tempptr = temp_result.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    if (!all_true)
                        continue;
                    tempptr[i] = op(a0, rightptr[i]);
                    if (!tempptr[i])
                    {
                        all_true = false;
                    }
                }
            }

            return all_true;
        }

        if (right.dims == 2)
        {
            // type 8
            temp_result.create(w1, h1, elemsize, opt.blob_allocator);
            if (temp_result.empty())
                return -100;
            bool all_true = true;
            const float* rightptr = right;
            bool* tempptr = temp_result;

            for (int y = 0; y < h1; y++)
            {
                if (!all_true)
                    continue;
                const float a0 = left[y];
                for (int x = 0; x < w1; x++)
                {
                    if (!all_true)
                        continue;
                    tempptr[x] = op(a0, rightptr[x]);
                    if (!tempptr[x])
                    {
                        all_true = false;
                    }
                }
                rightptr += w1;
                tempptr += w1;
            }

            return all_true;
        }

        if (right.dims == 1)
        {
            if (right.w == 1)
            {
                // type 6
                temp_result.create(w, elemsize, opt.blob_allocator);
                if (temp_result.empty())
                    return -100;
                bool all_true = true;
                const float b0 = right[0];
                for (int i = 0; i < w; i++)
                {
                    if (!all_true)
                        continue;
                    temp_result[i] = op(left[i], b0);
                    if (!temp_result[i])
                    {
                        all_true = false;
                    }
                }

                return all_true;
            }

            // type 7
            bool all_true = true;
            for (int i = 0; i < w; i++)
            {
                if (!all_true)
                    continue;
                temp_result[i] = op(left[i], right[i]);
                if (!temp_result[i])
                {
                    all_true = false;
                }
            }
            return all_true;
            //To do: compare with binaryop to check if this branch is right.

        }
    }

    return 0;
}


struct equal_op
{
    bool operator()(const float& x, const float& y) const
    {
        return abs(x - y) < EPS;
    }
};
struct less_op
{
    bool operator()(const float& x, const float& y) const
    {
        return x < y;
    }
};
struct greater_op
{
    bool operator()(const float& x, const float& y) const
    {
        return x > y;
    }
};
struct not_equal_op
{
    bool operator()(const float& x, const float& y) const
    {
        return abs(x - y) >= EPS;
    }
};
struct le_op
{
    bool operator()(const float& x, const float& y) const
    {
        return x <= y;
    }
};
struct ge_op
{
    bool operator()(const float& x, const float& y) const
    {
        return x >= y;
    }
};
int Compare::forward(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];



    if (op_type == Operation_EQUAL)
        return condition_compare<equal_op>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_LESS)
        return condition_compare<less_op>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_GREATER)
        return condition_compare<greater_op>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_NE)
        return condition_compare<not_equal_op>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_LE)
        return condition_compare<le_op>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_GE)
        return condition_compare<ge_op>(bottom_blob, bottom_blob1, top_blob, opt);

    return 0;
}


} // namespace ncnn
