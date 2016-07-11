#ifndef CAFFE_VISUAL_LAYER_HPP_
#define CAFFE_VISUAL_LAYER_HPP_

#include <stdio.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template <typename Dtype>
class VisualLayer : public Layer<Dtype> {
 public:

  explicit VisualLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      //vector<int> top_shape(0); 
      //top[0]->Reshape(top_shape);
  };

  virtual inline const char* type() const { return "Visual"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

  virtual inline int MinTopBlobs() const { return 0; }
  virtual inline int MaxTopBlos() const { return 0; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- VisualLayer 不需要回传梯度.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

 private:
  cv::Mat BlobToColorImage(const Blob<Dtype>* blob, const int n, const int height, const int width, const Dtype img_scale);
  cv::Mat BlobToGreyImage(const Blob<Dtype>* blob, const int n, const int height, const int width, const Dtype img_scale);
  void BlobToFourChannelImage(const Blob<Dtype>* blob, const int n, const int height, const int width, const Dtype img_scale);
  int iteration_num;
  int save_interval;
  string save_path;
  float scale;
};

}  // namespace caffe

#endif  // CAFFE_VISUAL_LAYER_HPP_
