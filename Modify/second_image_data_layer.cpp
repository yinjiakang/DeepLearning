#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/second_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
SecondImageDataLayer<Dtype>::~SecondImageDataLayer<Dtype>() {
    this->StopInternalThread();
}

template <typename Dtype>
void SecondImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        const int new_height = this->layer_param_.image_data_param().new_height();
        const int new_width  = this->layer_param_.image_data_param().new_width();
        //const bool is_color  = this->layer_param_.image_data_param().is_color();
        string root_folder = this->layer_param_.image_data_param().root_folder();

    CHECK((new_height == 0 && new_width == 0) ||
        (new_height > 0 && new_width > 0)) << "Current implementation requires "
            "new_height and new_width to be set at the same time.";
    // Read the file with filenames and labels
    const string& source = this->layer_param_.image_data_param().source();
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    string line;
    //size_t pos;
    //int label;

    //#############################
    // modified by Yinjk
    //#############################
    string original, label, prediction;

    while (infile >> original >> label >> prediction) {
        LOG(INFO) << original << " " << label << " " << prediction << "\n";
        lines_.push_back(std::make_pair(std::make_pair(original, prediction), label));
    }


    CHECK(!lines_.empty()) << "File is empty";

    if (this->layer_param_.image_data_param().shuffle()) {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        const unsigned int prefetch_rng_seed = caffe_rng_rand();
        prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
        ShuffleImages();
    }
    //LOG(INFO) << "A total of " << lines_.size() << " images.";

    lines_id_ = 0;
    // Check if we would need to randomly skip a few data points
    if (this->layer_param_.image_data_param().rand_skip()) {
        unsigned int skip = caffe_rng_rand() %
                this->layer_param_.image_data_param().rand_skip();
        LOG(INFO) << "Skipping first " << skip << " data points.";
        CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
        lines_id_ = skip;
    }

    //LOG(INFO) << "root_folder:  " << root_folder << "\n";
    //LOG(INFO) << "lines_[lines_id_].first.first:  " << lines_[lines_id_].first.first << "\n";

    // Read an image, and use it to initialize the top blob.


    //#############################
    // modified by Yinjk
    //#############################

    //LOG(INFO) << "lines_[lines_id_].first.second:  " << lines_[lines_id_].first.second << "\n";

  


    vector<int> top_shape(4);
    // Reshape prefetch_data and top[0] according to the batch_size.
    const int batch_size = this->layer_param_.image_data_param().batch_size();
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
    //LOG(INFO) << "lines_[lines_id_].second:  " << lines_[lines_id_].second << "\n";
   
    vector<int> label_shape(4);
    label_shape[0] = batch_size;
    label_shape[1] = 1;
    label_shape[2] = 128;
    label_shape[3] = 128;   
    top[1]->Reshape(label_shape); 


    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].label_.Reshape(label_shape);
    }

    LOG(INFO) << "output label size: " << top[1]->num() << ","
            << top[1]->channels() << "," << top[1]->height() << ","
            << top[1]->width();
}

template <typename Dtype>
void SecondImageDataLayer<Dtype>::ShuffleImages() {
    caffe::rng_t* prefetch_rng =
            static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void SecondImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    //CHECK(this->transformed_data_.count());

    ImageDataParameter image_data_param = this->layer_param_.image_data_param();
    const int batch_size = image_data_param.batch_size();
    const int new_height = image_data_param.new_height();
    const int new_width = image_data_param.new_width();
    const bool is_color = image_data_param.is_color();
    string root_folder = image_data_param.root_folder();

    // Reshape according to the first image of each batch
    // on single input batches allows for inputs of varying dimension.



    vector<int> top_shape(4);
    top_shape[0] = batch_size;
    top_shape[1] = 4;
    top_shape[2] = new_height;
    top_shape[3] = new_width;
    batch->data_.Reshape(top_shape);


    vector<int> label_shape(4);
    label_shape[0] = batch_size;
    label_shape[1] = 1;
    label_shape[2] = 128;
    label_shape[3] = 128;   
    batch->label_.Reshape(label_shape);


    Dtype* prefetch_data = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();

    // datum scales
    const int lines_size = lines_.size();
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        // get a blob
        timer.Start();
        CHECK_GT(lines_size, lines_id_);


        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
                new_height, new_width, is_color);
        CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first.first;
        

        cv::Mat cv_prediction = ReadImageToCVMat(root_folder + lines_[lines_id_].first.second,
                    new_height, new_width, false);
        CHECK(cv_prediction.data) << "Could not load " << lines_[lines_id_].first.second;


        cv::Mat cv_label = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                    128 , 128, false);
        CHECK(cv_label.data) << "Could not load " << lines_[lines_id_].second;


        read_time += timer.MicroSeconds();
        timer.Start();

        //#############################
        // modified by Yinjk
        //#############################
        

        for (int c = 0; c < 4; c++) {
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    int a;
                    if (c == 0) {
                        a = (int)cv_prediction.at<uchar>(i, j);
                    } else {
                        a = cv_img.at<cv::Vec3b>(i, j)[c];
                    }
                    //LOG(INFO) << a << "\n";
                    prefetch_data[item_id * 4 * 256 * 256 + c * 256 * 256 + i * 256 + j] = a;
                }
            }
        }



        // =============================================================================


        for (int i = 0; i < 128;  ++i) {
            for (int j = 0; j < 128; ++j) {
                int a = cv_label.at<uchar>(i, j);
                if (a == 0) {
                    prefetch_label[item_id * 128 * 128 + i * 128 + j] = Dtype(0);
                } else {
                    prefetch_label[item_id * 128 * 128 + i * 128 + j] = Dtype(1);
                }

            }
        }
        // =============================================================================


        //#############################
        // modified by Yinjk
        //#############################
        // Apply transformations (mirror, crop...) to the image
        
        //int offset = batch->data_.offset(item_id);
        //this->transformed_data_.set_cpu_data(prefetch_data + offset);
        //this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
        trans_time += timer.MicroSeconds();

        //prefetch_label[item_id] = lines_[lines_id_].second;

        
        //int offset = batch->label_.offset(item_id);
        //this->transformed_data_.set_cpu_data(prefetch_data + offset);
        //this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
        



        // go to the next iter
        lines_id_++;
        if (lines_id_ >= lines_size) {
            // We have reached the end. Restart from the first.
            DLOG(INFO) << "Restarting data prefetching from start.";
            lines_id_ = 0;
            if (this->layer_param_.image_data_param().shuffle()) {
                ShuffleImages();
            }
        }
    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(SecondImageDataLayer);
REGISTER_LAYER_CLASS(SecondImageData);

}  // namespace caffe
#endif  // USE_OPENCV
