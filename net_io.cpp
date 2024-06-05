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
}

int load_conv_bn_relu_layer(ConvBnRelu & layer, int offset, const vector<int> layer_header,vector<unsigned char> & buf){

  layer.inChannels=layer_header[1];
  layer.outChannels=layer_header[2];
  layer.kh = layer_header[3];
  layer.kw = layer_header[4];
  layer.strideX = layer_header[5];
  layer.strideY = layer_header[6];
  layer.padX = layer_header[7];
  layer.padY = layer_header[8];
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

void read_net(const char * path){
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
  while (i<n_layers){
    vector<int> layer_meta;
    copy_vector_int(buf, layer_meta, offset, H_LEN);
    offset+=H_LEN;
    if (layer_meta[0] == 1){ // convolution
      ConvBnRelu conv;
      offset = load_conv_bn_relu_layer(conv, offset, layer_meta, buf);
    }
    
  }
}

