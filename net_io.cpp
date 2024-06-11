#include <fstream>
#include <unordered_map> 
#include "bits/stdc++.h"
#include <iostream>
#include "utils.h"

#define H_LEN 256

using namespace std;

void copy_vector_int(const vector<unsigned char> & buf, 
    vector<int> & data, const int offset, const int size){

  //cout <<"int size " << size <<endl;
  data.resize(size);
  auto start=buf.begin()+offset*4;
  memcpy(data.data(), &(*start), data.size()*sizeof(int));
}

void copy_vector_float(const vector<unsigned char> & buf, vector<float> & data, 
    const int offset, const int size){
  data.resize(size);
  auto start=buf.begin()+offset*4;
  memcpy(data.data(), &(*start), data.size()*sizeof(float));

}

void read_ckpt(const char * path, Tensor & tensor){
  ifstream in(path, ios::binary);
  in.unsetf(ios::skipws);
  vector<unsigned char> buf; 
  in.seekg(0, ios::end);
  streampos size=in.tellg();
  in.seekg(0, ios::beg);
  buf.reserve(size);
  //cout <<"file size " << size <<endl;


  buf.insert(
    buf.begin(),
    istream_iterator<unsigned char> (in),
    istream_iterator<unsigned char> ()
  );

  vector<int> header;
  copy_vector_int(buf, header, 0, 256);
  
  tensor.b=header[2];
  tensor.c=header[3];
  tensor.h=header[4];
  tensor.w=header[5];
  int tensor_size=tensor.b*tensor.c*tensor.h*tensor.w;
  copy_vector_float(buf, tensor.data, 256, tensor_size); 
  in.close();
}

int load_conv_bn_relu_layer(ConvBnRelu & layer, int offset, vector<unsigned char> & buf){
  vector<int> layer_header;
  copy_vector_int(buf, layer_header, offset, H_LEN);
  if (layer_header[0] != 1){
    cout << "wrong header for conv, got " << layer_header[0] << " expected 1 "  << endl;
  }
  offset+=H_LEN;

  layer.inChannels=layer_header[1];
  layer.outChannels=layer_header[2];
  layer.kh = layer_header[3];
  layer.kw = layer_header[4];
  layer.strideX = layer_header[5];
  layer.strideY = layer_header[6];
  layer.padX = layer.padW = layer_header[7];
  layer.padY = layer.padH = layer_header[8];
  layer.doBias= layer_header[9];
  layer.group= layer_header[10];
  layer.dilationX = layer_header[11];
  layer.dilationY = layer_header[12];
  layer.kd = layer_header[13]; // layer shape 1 dim
  layer.kc = layer_header[14]; 
  layer.kh = layer_header[15]; 
  layer.kw = layer_header[16];
  layer.num_features=layer.kd;

  int tensor_size=layer.kd*layer.kc*layer.kh*layer.kw;
  copy_vector_float(buf, layer.convWeight, offset, tensor_size); 
  offset += tensor_size;
  if (layer.doBias){
    copy_vector_float(buf, layer.biasWeight, offset, layer.kd);
    offset += layer.kd;
  }

  vector<int> next_header;
  copy_vector_int(buf, next_header, offset, H_LEN);
  if (next_header[0] == 2){ // next is bn
    offset += H_LEN;
    layer.doBn=true;
    tensor_size=layer.kd;
    copy_vector_float(buf, layer.bnWeight, offset, tensor_size);
    offset+=layer.kd; 
    copy_vector_float(buf, layer.bnBias, offset, tensor_size);
    offset+=layer.kd; 
    copy_vector_float(buf, layer.bnRunningMean, offset, tensor_size);
    offset+=layer.kd; 
    copy_vector_float(buf, layer.bnRunningVar, offset, tensor_size);
    offset+=layer.kd; 
  }else{
    return offset;
  }

  copy_vector_int(buf, next_header, offset, H_LEN);
  if (next_header[0] == 3){ // relu
    offset += H_LEN;
    layer.doRelu = true;
  }

  return offset;
}

int load_max_pool(MaxPool & layer, int offset, vector<unsigned char> & buf ){
  vector<int> layer_header;
  copy_vector_int(buf, layer_header, offset, H_LEN);
  cout << "loading maxpool " << layer_header[5] << " expected 102 "  << endl;
  if (layer_header[0] != 4){
    return offset;
  }
  offset+=H_LEN;

  layer.kernel_size=layer_header[1];
  layer.stride=layer_header[2];
  layer.padding=layer_header[3];
  layer.dilation=layer_header[4];

  return offset;
}
void load_avg_pool(AvgPool & layer, int offset, vector<unsigned char> & buf){}
void load_linear(Linear & layer, int offset, vector<unsigned char> & buf){
  vector<int> layer_header;
  copy_vector_int(buf, layer_header, offset, H_LEN);
  offset+=H_LEN;
}

void read_net(const char * path, Resnet18 & model){
  ifstream in(path, ios::binary);
  in.unsetf(ios::skipws);
  vector<unsigned char> buf; 
  in.seekg(0, ios::end);
  streampos size=in.tellg();
  in.seekg(0, ios::beg);
  buf.reserve(size);


  buf.insert(
    buf.begin(),
    istream_iterator<unsigned char> (in),
    istream_iterator<unsigned char> ()
  );



  vector<int> net_meta;
  copy_vector_int(buf,net_meta, 0, H_LEN);
  int n_layers = net_meta[2];

  int offset=256;
  int i = 0;
  offset = load_conv_bn_relu_layer(model.conv1, offset,  buf);
  offset = load_max_pool(model.pool1, offset, buf);

  offset = load_conv_bn_relu_layer(model.l1_b0_conv1, offset, buf);
  offset = load_conv_bn_relu_layer(model.l1_b0_conv2, offset, buf);
  offset = load_conv_bn_relu_layer(model.l1_b1_conv1, offset, buf);
  offset = load_conv_bn_relu_layer(model.l1_b1_conv2, offset, buf);

  offset = load_conv_bn_relu_layer(model.l2_b0_conv1, offset, buf);
  offset = load_conv_bn_relu_layer(model.l2_b0_conv2, offset, buf);
  offset = load_conv_bn_relu_layer(model.l2_b0_conv3, offset, buf);
  offset = load_conv_bn_relu_layer(model.l2_b1_conv1, offset, buf);
  offset = load_conv_bn_relu_layer(model.l2_b1_conv2, offset, buf);

  offset = load_conv_bn_relu_layer(model.l3_b0_conv1, offset, buf);
  offset = load_conv_bn_relu_layer(model.l3_b0_conv2, offset, buf);
  offset = load_conv_bn_relu_layer(model.l3_b0_conv3, offset, buf);
  offset = load_conv_bn_relu_layer(model.l3_b1_conv1, offset, buf);
  offset = load_conv_bn_relu_layer(model.l3_b1_conv2, offset, buf);

  offset = load_conv_bn_relu_layer(model.l4_b0_conv1, offset, buf);
  offset = load_conv_bn_relu_layer(model.l4_b0_conv2, offset, buf);
  offset = load_conv_bn_relu_layer(model.l4_b0_conv3, offset, buf);
  offset = load_conv_bn_relu_layer(model.l4_b1_conv1, offset, buf);
  offset = load_conv_bn_relu_layer(model.l4_b1_conv2, offset, buf);


  in.close();
  cout << "read net ok " << endl; 
}

