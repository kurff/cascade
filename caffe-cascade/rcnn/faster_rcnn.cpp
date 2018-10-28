#include "rcnn/faster_rcnn.h"

using namespace caffe;
namespace rcnn{
    FasterRCNN::FasterRCNN(float threshold) : FastRCNN(threshold){

    }

    FasterRCNN::~FasterRCNN(){

    }

    

    bool FasterRCNN::forward(cv::Mat& image, std::vector<BBox>* boxes){
        Blob<float>* input_layer = net_->input_blobs()[0];
        float orgH = float(image.rows);
        float orgW = float(image.cols);
        float rzRatio = float(shortSize_)/ std::min(orgH, orgW);
        int imgH = int(min(rzRatio*orgH,longSize_))/32*32;
        int imgW = int(min(rzRatio*orgW,longSize_))/32*32;

        float hratio = float(imgH)/orgH;
        float wratio = float(imgW)/orgW;

        input_geometry_.height = imgH;
        input_geometry_.width = imgW;

        input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);
            /* Forward dimension change to all layers. */
        net_->Reshape();
        std::vector<cv::Mat> input_channels;
        WrapInputLayer(&input_channels);

        Preprocess(image, &input_channels);

        net_->ForwardPrefilled();

            /* Copy the output layer to a std::vector */
        Blob<float>* output_layer_cls = net_->output_blobs()[0];
        Blob<float>* output_layer_box = net_->output_blobs()[1];

        LOG(INFO)<<"box shape: "<< output_layer_box->shape_string();
        LOG(INFO)<< "cls shape: "<<output_layer_cls->shape_string();
        int box_dim = output_layer_box->count(1);
        int cls_dim = output_layer_cls->count(1);
        const float* box_data = output_layer_box->cpu_data();
        const float* cls_data = output_layer_cls->cpu_data();
        

        boxes->clear();
        vector<BBox> all_boxes;
        for(int i = 0; i < output_layer_box->num(); ++ i){
            float score = cls_data[1];
            LOG(INFO)<<"cls: "<< score;
            //LOG(INFO)<<"box: "<< box_data[0]<<" "<< box_data[1]<<" "
            //<<box_data[2]<<" "<< box_data[3]<<" "<< box_data[4];
            if( score >= this->threshold_ ){
                BBox box;
                box.xmin = std::max<float>(box_data[1]/wratio,0.0f);
                box.ymin = std::max<float>(box_data[2]/hratio,0.0f);
                box.xmax = std::min<float>(box_data[3]/wratio,orgW);
                box.ymax = std::min<float>(box_data[4]/hratio,orgH);
                box.score = score;
                all_boxes.push_back(box);
            }
            box_data += box_dim;
            cls_data += cls_dim;
        }

        std::sort(all_boxes.begin(), all_boxes.end(), SortBBoxDescend);
        *boxes = ApplyNMS(all_boxes, 0.5, 1, "IOMU");

        LOG(INFO)<<"height: "<< input_geometry_.height<<" width: "<< input_geometry_.width; 
        return true;
    }

    bool FasterRCNN::forward(cv::Mat& image, const std::vector<BBox>& prev_boxes, std::vector<BBox>* curr_boxes){


    }

    
} // end of namespace rcnn