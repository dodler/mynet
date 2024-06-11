#ifndef COMMON_H_
#define COMMON_H_

#include <vector>
#include <variant>

using namespace std;


struct Tensor{
  int b,c,h,w;
  vector<float> data;
}; 

struct Layer{
  public: 
    int layer_type=-1;
};

struct ConvBnRelu {
  public:
    int padX=0,padY=0,strideX=1, strideY=1,padH=0,padW=0,group=1;
    int dilationX=1,dilationY=1;
    int inChannels=-1,outChannels=-1;
    int kh,kc,kw,kd;
    bool doBias=false,doRelu=false,doBn=false;
    int num_features=-1;

    vector<float> convWeight;
    vector<float> biasWeight;
    vector<float> bnRunningMean;
    vector<float> bnRunningVar;
    vector<float> bnWeight;
    vector<float> bnBias;
};

struct MaxPool {
  public:
    int kernel_size, stride, padding, dilation;
};

struct AvgPool {
  public:
    int size;
};

struct Linear{
  public:
    int in_features, out_features;
    vector<float> weight, bias;
};

struct Resnet18{
  ConvBnRelu conv1;
  MaxPool pool1;
  ConvBnRelu l1_b0_conv1, l1_b0_conv2, l1_b1_conv1, l1_b1_conv2;
  ConvBnRelu l2_b0_conv1, l2_b0_conv2, l2_b0_conv3, l2_b1_conv1, l2_b1_conv2;
  ConvBnRelu l3_b0_conv1, l3_b0_conv2, l3_b0_conv3, l3_b1_conv1, l3_b1_conv2;
  ConvBnRelu l4_b0_conv1, l4_b0_conv2, l4_b0_conv3, l4_b1_conv1, l4_b1_conv2;

  AvgPool pool;
  Linear fc;
};

#endif
