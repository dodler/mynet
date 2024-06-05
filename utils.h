#ifndef COMMON_H_
#define COMMON_H_

#include <vector>

using namespace std;

struct Tensor{
  int b,c,h,w;
  vector<float> data;
}; 


struct ConvBnRelu{
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

#endif
