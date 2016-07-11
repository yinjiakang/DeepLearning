#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/visual_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void VisualLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
  iteration_num = 0;
  save_interval = this->layer_param_.visual_param().save_interval();
  save_path = this->layer_param_.visual_param().save_folder();
  scale = this->layer_param_.visual_param().scale();

  CHECK_GT(save_interval, 0);
}



template <typename Dtype>
void VisualLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ++iteration_num;
  if(iteration_num % save_interval != 0)
    return;

  char name_buffer[100];
  cv::Mat rgb_img;
  cv::Mat label_img;
  cv::Mat prediction_img;

  for(int img_id = 0; img_id < bottom[0]->num(); ++img_id) {
    //rgb_img        = BlobToColorImage(bottom[0], img_id, bottom[0]->height(), bottom[0]->width(), Dtype(1));
    BlobToFourChannelImage(bottom[0], img_id, bottom[0]->height(), bottom[0]->width(), scale);
    label_img      = BlobToGreyImage(bottom[1], img_id, bottom[1]->height(), bottom[1]->width(), scale);
    prediction_img = BlobToGreyImage(bottom[2], img_id, bottom[1]->height(), bottom[1]->width(), scale);

    //label_img = BlobToGreyImage(bottom[1], img_id, scale);
    //prediction_img = BlobToGreyImage(bottom[2], img_id, scale);


    sprintf(name_buffer, "%d_%d_label.jpg", iteration_num, img_id);
    cv::imwrite(save_path + string(name_buffer), label_img);

    sprintf(name_buffer, "%d_%d_prediction.jpg", iteration_num, img_id);
    cv::imwrite(save_path + string(name_buffer), prediction_img);
  }
}


template <typename Dtype>
cv::Mat VisualLayer<Dtype>::BlobToColorImage(const Blob<Dtype>* blob, const int n, 
  const int height, const int width, const Dtype img_scale) {
  CHECK_GE(blob->channels(), 3) << "Only Support Color images";

  const Dtype* cpu_data = blob->cpu_data();
  int offset = n * blob->channels() * height * width;

  cv::Mat img(height, width, CV_8UC3);
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < img.rows; ++h) {
      for (int w = 0; w < img.cols; ++w) {
        Dtype value = cpu_data[offset] * img_scale;
        if(value < 0) value = 0;
        if(value > 255) value = 255;
        
        img.at<cv::Vec3b>(h, w)[c] = cv::saturate_cast<uchar>(value);

        ++offset;
      }
    }
  }

  return img;
}

template <typename Dtype>
cv::Mat VisualLayer<Dtype>::BlobToGreyImage(const Blob<Dtype>* blob, const int n, 
  const int height, const int width, const Dtype img_scale) {
  CHECK_GE(blob->channels(), 1) << "Only Support grey images";

 const Dtype* cpu_data = blob->cpu_data();
 int offset = n * height * width;

  cv::Mat img(height, width, CV_8UC1);
  for (int h = 0; h < img.rows; ++h) {
    for (int w = 0; w < img.cols; ++w) {
      //Dtype value = blob->data_at(n, 0, h, w) * img_scale;
      Dtype value = cpu_data[offset] * img_scale;
      if(value < 0) value = 0;
      if(value > 255) value = 255;

      img.at<uchar>(h, w) = cv::saturate_cast<uchar>(value);
      ++offset;
    }
  }

  return img;
}

template <typename Dtype>
void VisualLayer<Dtype>::BlobToFourChannelImage(const Blob<Dtype>* blob, const int n, 
  const int height, const int width, const Dtype img_scale) {
  CHECK_GE(blob->channels(), 4) << "Only Support Four channel images";

  const Dtype* cpu_data = blob->cpu_data();
  int offset = n * blob->channels() * height * width;
  char name_buffer[100];

  //LOG(INFO) << height << width << "\n";

  cv::Mat img(height, width, CV_8UC3);
  cv::Mat img2(height, width, CV_8UC1);
  for (int c = 0; c < 4; ++c) {
    for (int h = 0; h < img.rows; ++h) {
      for (int w = 0; w < img.cols; ++w) {
        Dtype value = cpu_data[offset];
        if(value < 0) value = 0;
        if(value > 255) value = 255;
        
        if (c == 0) {
          img2.at<uchar>(h, w) = cv::saturate_cast<uchar>(value);
        } else {
          img.at<cv::Vec3b>(h, w)[c] = cv::saturate_cast<uchar>(value);
        }

        ++offset;
      }
    }
  }

  sprintf(name_buffer, "%d_%d_img_0.jpg", iteration_num, n);
  cv::imwrite(save_path + string(name_buffer), img);

  sprintf(name_buffer, "%d_%d_img_1.jpg", iteration_num, n);
  cv::imwrite(save_path + string(name_buffer), img2);

}

INSTANTIATE_CLASS(VisualLayer);
REGISTER_LAYER_CLASS(Visual);

}  // namespace caffe
