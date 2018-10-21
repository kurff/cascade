#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include "caffe/util/io.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/text_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
TextDataLayer<Dtype>::TextDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
TextDataLayer<Dtype>::~TextDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void TextDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  AnnoDatum anno_datum;
  anno_datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> box_shape(4, batch_size);
    box_shape[1] = 1;
    box_shape[2] = 1;
    box_shape[3] = 5;
    top[1]->Reshape(box_shape);

    for (int i = 0; i < this->prefetch_.size(); ++i) {
      shared_ptr<Blob<Dtype> > label_blob_pointer(new Blob<Dtype>());
      label_blob_pointer->Reshape(box_shape);
      this->prefetch_[i]->labels_.push_back(label_blob_pointer);
    }  
  }
}

template <typename Dtype>
bool TextDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void TextDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void TextDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  AnnoDatum anno_datum;

  vector<std::shared_ptr<Blob<Dtype> > > bboxes;
  bboxes.clear();
  for(int item_id = 0; item_id < batch_size; ++ item_id){
    std::shared_ptr<Blob<Dtype> > ptr(new Blob<Dtype>());
    bboxes.push_back(ptr);
  }

  int total_boxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    anno_datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    Datum datum = anno_datum.datum();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);

    Dtype* top_data = batch->data_.mutable_cpu_data();
    cv::Mat cv_img = DecodeDatumToCVMat(datum, 1);

    //const int number = batch->data_.num();
    //const int channels = batch->data_.channels();
    const int height = batch->data_.height();
    const int width = batch->data_.width();

    for(int i = 0; i < cv_img.rows; ++i){
      for(int j = 0; j < cv_img.cols; ++ j){
        cv::Vec3b v = cv_img.at<cv::Vec3b>(i,j);
        top_data[offset + i*width +j] = static_cast<Dtype>( static_cast<uint8_t>( v.val[0]));
        top_data[offset + height*width + i * width + j] = static_cast<Dtype>( static_cast<uint8_t>( v.val[1]));
        top_data[offset + 2*height*width + i * width + j] = static_cast<Dtype>( static_cast<uint8_t>( v.val[2]));
      }
    }
    // Copy label.
    if (this->output_labels_) {
      const int num_box = anno_datum.bbox_size();
      vector<int> box_shape(4, 1);
      box_shape[2] = num_box;
      box_shape[3] = 5;
      bboxes[item_id]->Reshape(box_shape);
      Dtype* bboxes_data = bboxes[item_id]->mutable_cpu_data();
      for(int i = 0; i < num_box; ++ i){
        bboxes_data[i*5 ] = item_id;
        bboxes_data[i*5+1] = anno_datum.bbox(i).xmin();
        bboxes_data[i*5+2] = anno_datum.bbox(i).ymin();
        bboxes_data[i*5+3] = anno_datum.bbox(i).xmax();
        bboxes_data[i*5+4] = anno_datum.bbox(i).ymax();
      }

      total_boxes+= num_box;
    }
    trans_time += timer.MicroSeconds();
    Next();
  }


  vector<int> label_shape(4,1);
  label_shape[2] = total_boxes;
  label_shape[3] = 5;

  batch->labels_[0]->Reshape(label_shape);
  Dtype* top_label = batch->labels_[0]->mutable_cpu_data();
  int count = 0;
  for(size_t i = 0; i < bboxes.size(); ++ i){
    for(int j = 0; j < bboxes[i]->channels(); ++ j){
      top_label[count*5] = bboxes[i]->data_at(0,0,j,0);
      top_label[count*5+1] = bboxes[i]->data_at(0,0,j,1);
      top_label[count*5+2] = bboxes[i]->data_at(0,0,j,2);
      top_label[count*5+3] = bboxes[i]->data_at(0,0,j,3);
      top_label[count*5+4] = bboxes[i]->data_at(0,0,j,4);
      ++ count;
    }
  }


  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(TextDataLayer);
REGISTER_LAYER_CLASS(TextData);

}  // namespace caffe
