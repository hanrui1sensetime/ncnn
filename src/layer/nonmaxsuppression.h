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

#ifndef LAYER_NONMAXSUPPRESSION_H
#define LAYER_NONMAXSUPPRESSION_H

#include "layer.h"

namespace ncnn {

class NonMaxSuppression : public Layer
{
public:
    NonMaxSuppression();

    //virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector <Mat>& bottom_blobs, Mat& top_blob, const Option& opt) const;
//private:
    float calculate_iou (float x1, float y1, float x2, float y2, 
                              float target_x1, float target_y1, float target_x2, float target_y2) const;
    float calculate_rectangle_area (float x1, float y1, float x2, float y2) const;
    

};

} // namespace ncnn

#endif // LAYER_NONMAXSUPPRESSION_H
