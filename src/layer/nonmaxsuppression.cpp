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

#include "nonmaxsuppression.h"

#include <algorithm>

namespace ncnn {

NonMaxSuppression::NonMaxSuppression()
{
    one_blob_only = false;
    support_inplace = false;
}
float NonMaxSuppression::calculate_rectangle_area (float x1, float y1, float x2, float y2) const
{
    return abs((x2 - x1) * (y2 - y1));
}
float NonMaxSuppression::calculate_iou (float x1, float y1, float x2, float y2, float target_x1, float target_y1, float target_x2, float target_y2) const
{
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    if (target_x1 > target_x2) std::swap(target_x1, target_x2);
    if (target_y1 > target_y2) std::swap(target_y1, target_y2);
    

    float intersect_x1 = std::max(target_x1, x1);
    float intersect_x2 = std::min(target_x2, x2);
    float intersect_y1 = std::max(target_y1, y1);
    float intersect_y2 = std::min(target_y2, y2);
    float intersect_w = std::max((float) 0.0, intersect_x2 - intersect_x1);
    float intersect_h = std::max((float) 0.0, intersect_y2 - intersect_y1);
    // To do: check if the calculation of intersect_w is right, and if the union_xx is proper. 

    float intersection = intersect_w * intersect_h;
    float area = NonMaxSuppression::calculate_rectangle_area(x1, y1, x2, y2);
    float target_area = NonMaxSuppression::calculate_rectangle_area(target_x1, target_y1, target_x2, target_y2);

    float iou = intersection / (area + target_area - intersection);
    return iou;
    // Should to change to parallel?

}

int NonMaxSuppression::forward(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Option& opt) const
{
    const Mat& bottom_boxes_blob = bottom_blobs[0];
    const Mat& bottom_scores_blob = bottom_blobs[1];
    const Mat& bottom_maxbox_perclass_blob = bottom_blobs[2];
    // How to reflect "per class"?
    const Mat& bottom_iou_threshold_blob = bottom_blobs[3];
    const Mat& bottom_score_threshold_blob = bottom_blobs[4];

    //const float* box_ptr = bottom_boxes_blob;
    const float* score_ptr = bottom_scores_blob;
    int size = bottom_boxes_blob.total();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(score_ptr[i], i);
    }
    std::sort(vec.begin(),vec.end(),std::greater<std::pair<float, int> >());

    const int* max_boxes_ptr = bottom_maxbox_perclass_blob;
    const float* iou_threshold_ptr = bottom_iou_threshold_blob;
    const float* score_threshold_ptr = bottom_score_threshold_blob;

    int max_boxes = max_boxes_ptr[0];
    float iou_threshold = iou_threshold_ptr[0];
    float score_threshold = score_threshold_ptr[0];
    

    std::vector<std::pair<float, int> > candidate_vec(vec);
    std::vector <int> indices;



    while (candidate_vec.size() > 0)
    {
        int best_index = candidate_vec[0].second;
        float score = candidate_vec[0].first;
        if (score < score_threshold)
        {
            break;
        }
        if (indices.size() < max_boxes) 
            indices.push_back(best_index);
        else
            break;
        
        int cur_candidate_vec_index = 0;
        for (int i = 1; i < candidate_vec.size(); i++)
        {
            int loop_index = candidate_vec[i].second;
            const float* loop_index_ptr = bottom_boxes_blob.row(loop_index);
            const float* best_index_ptr = bottom_boxes_blob.row(best_index);
            float iou = NonMaxSuppression::calculate_iou(loop_index_ptr[0], 
                                                         loop_index_ptr[1], 
                                                         loop_index_ptr[2], 
                                                         loop_index_ptr[3], 
                                                         best_index_ptr[0],
                                                         best_index_ptr[1], 
                                                         best_index_ptr[2],
                                                         best_index_ptr[3]);
            // To do: check the parameter order, if it is x1, y1, x2, y2?
            if (iou < iou_threshold)
            {
                candidate_vec[cur_candidate_vec_index] = candidate_vec[loop_index];
                cur_candidate_vec_index++;
            }
            // To do: check if the loop will mess up?
        }
        candidate_vec.resize(cur_candidate_vec_index);

    }

    // To do: Allocate the space for top_blob.
    int* outptr = top_blob;
    size_t elemsize = bottom_maxbox_perclass_blob.elemsize;
    top_blob.create(indices.size(), elemsize, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    for (int i = 0; i < indices.size(); i++)
    {
        top_blob[i] = indices[i];
    }

    return 0;
}

} // namespace ncnn
