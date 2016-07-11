#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/four_channel_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
FourChannelDataLayer<Dtype>::~FourChannelDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void FourChannelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.four_channel_data_param().new_height();
  const int new_width  = this->layer_param_.four_channel_data_param().new_width();
  const int label_height = this->layer_param_.four_channel_data_param().label_height();
  const int label_width  = this->layer_param_.four_channel_data_param().label_width();
  //const bool is_color  = this->layer_param_.four_channel_data_param().is_color();
  string root_folder = this->layer_param_.four_channel_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.four_channel_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos, pos1;
  while (std::getline(infile, line)) {
    std::vector<std::string> temp(3);
    pos = line.find_first_of(' ');
    pos1 = line.find_first_of(' ', pos + 1);
    temp[0] = line.substr(0, pos);
    temp[1] = line.substr(pos + 1, pos1 - pos - 1);
    temp[2] = line.substr(pos1 + 1);
    lines_.push_back(temp);
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.four_channel_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.four_channel_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.four_channel_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  // cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
  //                                   new_height, new_width, is_color);
  // CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape(4);
  // this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.four_channel_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  top_shape[1] = 4;
  top_shape[2] = new_height;
  top_shape[3] = new_width;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(4);
  //top[1]->Reshape(label_shape);
  // chaning label shapes ###############################################################################
  // cv::Mat cv_img1 = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
  //                                   64, 64, false);
  // CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].second;
  // vector<int> label_Shape = this->data_transformer_->InferBlobShape(cv_img1);
  // this->transformed_data_.Reshape(label_Shape);
  label_shape[0] = batch_size;
  label_shape[1] = 1;
  label_shape[2] = label_height;
  label_shape[3] = label_width;
  top[1]->Reshape(label_shape);
  LOG(INFO) << "output data size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
  // this is the end of chaninge#########################################################################
  
  // set the size of the blob of the labels
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void FourChannelDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void FourChannelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  //CHECK(this->transformed_data_.count());
  FourChannelDataParameter four_channel_data_param = this->layer_param_.four_channel_data_param();
  const int batch_size = four_channel_data_param.batch_size();
  const int new_height = four_channel_data_param.new_height();
  const int new_width = four_channel_data_param.new_width();
  const int label_height = four_channel_data_param.label_height();
  const int label_width = four_channel_data_param.label_width();
  const float scale = four_channel_data_param.scale();
  //const bool is_color = four_channel_data_param.is_color();
  string root_folder = four_channel_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  // cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
  //     new_height, new_width, true);
  // CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];
  // cv::Mat cv_img_predict = ReadImageToCVMat(root_folder + lines_[lines_id_][1],
  //     new_height, new_width, false);
  // CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][1];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape(4);
  //this->transformed_data_.Reshape(top_shape);
  
  // USe data_transformer to infer the expected blob shape from a cv_img, this is for multilabel layers####################
  // cv::Mat cv_img1 = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
  //     64, 64, false);
  // CHECK(cv_img1.data) << "Could not load " << lines_[lines_id_].second;
  
  vector<int> label_Shape(4);
  //this->transformed_data_.Reshape(label_Shape);
  
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  top_shape[1] = 4;
  top_shape[2] = new_height;
  top_shape[3] = new_width;
  batch->data_.Reshape(top_shape);
  
  // adding multilabel layers ####################################################################################
  label_Shape[0] = batch_size;
  label_Shape[1] = 1;
  label_Shape[2] = label_height;
  label_Shape[3] = label_width;
  batch->label_.Reshape(label_Shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    

    
    
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
        new_height, new_width, true);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];
    cv::Mat cv_img_predict = ReadImageToCVMat(root_folder + lines_[lines_id_][1],
        new_height, new_width, false);
    CHECK(cv_img_predict.data) << "Could not load " << lines_[lines_id_][1];
    cv::Mat cv_img1 = ReadImageToCVMat(root_folder + lines_[lines_id_][2], 
        label_height, label_width, false);
    CHECK(cv_img1.data) << "Could not load " << lines_[lines_id_][2];
    // vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    // this->transformed_data_.Reshape(top_shape);
    
    read_time += timer.MicroSeconds();
    timer.Start();

    // import four channel data
    // int offset = batch->data_.offset(item_id);
    // this->transformed_data_.set_cpu_data(prefetch_data + offset);
    // this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    int temp;
    int offset = item_id * 4 * new_height * new_width;
    for (int c = 0; c < 4; ++c) {
      for (int h = 0; h < new_height; ++h) {
        for (int w = 0; w < new_width; ++w) {
          if (c < 3) {
            temp = (int)cv_img.at<cv::Vec3b>(h, w)[c];
            
          }
          else {
            temp = (int)cv_img_predict.at<uchar>(h, w);
          }
          prefetch_data[offset] = temp;
          ++offset;
        }
      }
    }
    offset = item_id * label_height * label_width;
    //import label data
    for (int i = 0; i < label_height; ++i) {
      for (int j = 0; j < label_width; ++j) {
        temp = (int)cv_img1.at<uchar>(i, j);
        if (temp > 0) {
          temp = scale;
        }
        prefetch_label[offset] = temp / scale;
        ++offset;
      }
    }
    trans_time += timer.MicroSeconds();
    //prefetch_label[item_id] = lines_[lines_id_].second;
    
    
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.four_channel_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(FourChannelDataLayer);
REGISTER_LAYER_CLASS(FourChannelData);

}  // namespace caffe
#endif  // USE_OPENCV