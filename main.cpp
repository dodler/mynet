#include <iostream>
#include <vector>
#include "net_io.cpp"
#include <cmath>
#include "utils.h"

using namespace std;
float relu(float value){
  return value > 0 ? value : 0;
}

void relu (Tensor & src){
  for (int i = 0; i<src.b*src.c*src.h*src.w; i++){
    src.data[i] = src.data[i] > 0 ? src.data[i] : 0;
  }
}

void im2col(const float * src, int srcC, int srcH, int srcW,
    int kernelY, int kernelX, int dilationY, int dilationX,
    int strideY, int strideX, int padY, int padX, int padH, int padW, float * buf)
{
  int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
  int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
  int ind = 0;
  for (int sc = 0; sc < srcC; ++sc)
  {
    for (int ky = 0; ky < kernelY; ky++)
    {
      for (int kx = 0; kx < kernelX; kx++)
      {
        for (int dy = 0; dy < dstH; ++dy)
        {
          for (int dx = 0; dx < dstW; ++dx)
          {
            int sy = dy * strideY + ky * dilationY - padY;
            int sx = dx * strideX + kx * dilationX - padX;
            if (sy >= 0 && sy < srcH && sx >= 0 && sx < srcW)
              *buf++ = src[(sc*srcH + sy)*srcW + sx];
            else
              *buf++ = 0;
          }
        }
      }
    }
  }
}

void gemm_nn(int M, int N, int K, float alpha, const float * A, int lda,
    float beta, const float * B, int ldb, float * C, int ldc)
{
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      C[i*ldc + j] = beta;
      for (int k = 0; k < K; ++k)
        C[i*ldc + j] += alpha * A[i*lda + k] * B[k*ldb + j];
    }
  }
}


void linear(const Tensor & src, const Linear & layer, Tensor & dst){
  dst.data.reserve(layer.out_features*src.b);
  int M = src.b, K=layer.in_features, N=layer.out_features;

  gemm_nn(M, N, K, 1, src.data.data(), K,0,layer.weight.data(),N, dst.data.data(), N);

  for (int i = 0; i<layer.out_features; i++){
    dst.data[i] += layer.bias[i];
  }
}


void conv_bn_relu(const Tensor & input_tensor, const ConvBnRelu & layer, Tensor & result)
{
  int srcH=input_tensor.h, srcW=input_tensor.w;
  int padY=layer.padY, padX=layer.padX, dilationX=layer.dilationX, dilationY=layer.dilationY;
  int kernelY=layer.kh, kernelX=layer.kw, strideY=layer.strideY, strideX=layer.strideX;
  int padH=layer.padH, padW=layer.padW;
  int batch = input_tensor.b, group=layer.group, srcC=input_tensor.c;

  int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
  int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;

  int dstC=layer.kd;
  int dst_size=dstC*input_tensor.b*dstH*dstW;

  int sy = dstH * strideY +kernelY* dilationY + padY;
  int sx = dstW * strideX + kernelX * dilationX + padX;
  
  int buf_size = dstH*dstW*kernelX*kernelY*srcC;

  vector<float> buf;
  buf.reserve(buf_size);

  result.data.reserve(dst_size);
  result.b=batch;
  result.c=dstC;
  result.h=dstH;
  result.w=dstW;

  int M = dstC / group;
  int N = dstH * dstW;
  int K = srcC * kernelY * kernelX / group;

  float * dst = result.data.data();
  const float * src = input_tensor.data.data();
  const float eps = 1e-5;

  for (int b = 0; b < batch; ++b)
  {
    im2col(src, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX,
      strideY, strideX, padY, padX, padH, padW, buf.data());
    for (int g = 0; g < group; ++g)
      gemm_nn(M, N, K, 1, layer.convWeight.data() + M * K * g, K, 0, buf.data() + N * K * g, N, dst + M * N * g, N);

    for (int i = 0; i < dstC; ++i){
      float mean,var,alpha,beta;
      if (layer.doBn){
        mean=layer.bnRunningMean[i];
        var = sqrt(layer.bnRunningVar[i]+eps);
        alpha = layer.bnWeight[i];
        beta = layer.bnBias[i];
      }
      for (int j = 0; j < N; ++j){
        float x = dst[i*N + j];
        if (layer.doBias){
	  x+=layer.biasWeight[i];
	}
	if (layer.doBn){
          x = (x-mean) / var; 
	  x = x * alpha;
	  x = x + beta;
	}
	if (layer.doRelu == 1){
	  x=relu(x);
	}
	dst[i*N+j]=x;
      }
    }
    src += srcC*srcH*srcW;
    dst += dstC*dstH*dstW;
  }
}

void maxpool(Tensor & input_tensor, MaxPool & layer, Tensor & output )
{
  int dstH = (input_tensor.h + layer.padding* 2- (layer.dilation * (layer.kernel_size - 1) + 1)) / layer.stride + 1;
  int dstW = (input_tensor.w + layer.padding*2 - (layer.dilation * (layer.kernel_size - 1) + 1)) / layer.stride + 1;

  int dst_size = dstH*dstW*input_tensor.b*input_tensor.c;
  output.data.reserve(dst_size);
  float * dst = output.data.data();
  float* src = input_tensor.data.data();

  output.b=input_tensor.b;
  output.c=input_tensor.c;
  output.h=dstH;
  output.w=dstW;

  int dstC = input_tensor.c, srcC=input_tensor.c, srcW=input_tensor.w, srcH=input_tensor.h;

  for (int b = 0; b < input_tensor.b; ++b)
  {
      for (int dc = 0; dc < dstC; ++dc)
      {
        for (int dy = 0; dy < dstH; ++dy)
        {
          for (int dx = 0; dx < dstW; ++dx)
          {
            float m=-10000.0;
              for (int ky = 0; ky < layer.kernel_size; ky++)
              {
                for (int kx = 0; kx < layer.kernel_size; kx++)
                {
                  int sy = dy * layer.stride + ky * layer.dilation - layer.padding;
                  int sx = dx * layer.stride + kx * layer.dilation - layer.padding;
                  if (sy >= 0 && sy < srcH && sx >= 0 && sx < srcW){
                    float val = src[((b*srcC+dc)*srcH + sy)*srcW + sx];
		    if (val > m){
	              m=val;
		    }
		  }
                }
              }
            dst[((b*dstC + dc)*dstH + dy)*dstW + dx] = m;
          }
        }
      }
  }
}

void avgpool(Tensor & input_tensor, AvgPool & layer, Tensor & output )
{

  int dstH=1,dstW=1;
  int dst_size = dstH*dstW*input_tensor.b*input_tensor.c;
  output.data.reserve(dst_size);
  float * dst = output.data.data();
  float* src = input_tensor.data.data();

  output.b=input_tensor.b;
  output.c=input_tensor.c;
  output.h=dstH;
  output.w=dstW;

  float ksize=layer.size * layer.size;

  int dstC = input_tensor.c, srcC=input_tensor.c, srcW=input_tensor.w, srcH=input_tensor.h;

  for (int b = 0; b < input_tensor.b; ++b)
  {
      for (int dc = 0; dc < dstC; ++dc)
      {
        for (int dy = 0; dy < dstH; ++dy)
        {
          for (int dx = 0; dx < dstW; ++dx)
          {
            float m=0;
              for (int ky = 0; ky < layer.size; ky++)
              {
                for (int kx = 0; kx < layer.size; kx++)
                {
                  int sy = dy  + ky; 
                  int sx = dx  + kx;
                  m+= src[((b*srcC+dc)*srcH + sy)*srcW + sx];
                }
              }
            dst[((b*dstC + dc)*dstH + dy)*dstW + dx] = m/ksize;
          }
        }
      }
  }
}

void print_shape(const Tensor & t){
  cout << "tensor shape " << t.b << ","<<t.c<<","<<t.w<<","<<t.h<<endl;
}

void add(Tensor & src, Tensor & dst){
  for(int i = 0; i<src.b*src.c*src.h*src.w; i++){
    src.data[i] += dst.data[i];
  }
}

void print_weights(const ConvBnRelu & layer){
  cout << "conv weights 0:20" << endl;
  for (int i = 0; i<20;i++){
    cout << layer.convWeight[i] << " ";
  }
  cout << endl;
}

void basic_block_downsample(const ConvBnRelu & l1, const ConvBnRelu & l2, const ConvBnRelu & downsample, Tensor & src, Tensor & dst){
  Tensor x1;
  conv_bn_relu(src, l1, x1);
  conv_bn_relu(x1, l2, dst);

  Tensor id;
  conv_bn_relu(src, downsample, id);
  add(dst, id);

  relu(dst);

}

void basic_block(const ConvBnRelu & l1, const ConvBnRelu & l2, Tensor & src, Tensor & dst){
  Tensor x1;
  conv_bn_relu(src, l1, x1);
  conv_bn_relu(x1, l2, dst);

  add(dst, src);
  relu(dst);

}

int main(){
  Tensor img;
  read_ckpt("test_img.bin", img);

  Resnet18 model;
  read_net("r18.bin", model);

  Tensor x1;
  conv_bn_relu(img, model.conv1, x1);
  Tensor x2;
  maxpool(x1, model.pool1, x2);

  Tensor bb1,bb2;
  basic_block(model.l1_b0_conv1, model.l1_b0_conv2, x2, bb1); 
  basic_block(model.l1_b1_conv1, model.l1_b1_conv2, bb1, bb2); 

  Tensor bb3, bb4;
  basic_block_downsample(model.l2_b0_conv1, model.l2_b0_conv2, model.l2_b0_conv3, bb2, bb3); 
  basic_block(model.l2_b1_conv1, model.l2_b1_conv2, bb3, bb4); 

  Tensor bb5, bb6;
  basic_block_downsample(model.l3_b0_conv1, model.l3_b0_conv2, model.l3_b0_conv3, bb4, bb5); 
  basic_block(model.l3_b1_conv1, model.l3_b1_conv2, bb5, bb6); 

  Tensor bb7, bb8;
  basic_block_downsample(model.l4_b0_conv1, model.l4_b0_conv2, model.l4_b0_conv3, bb6, bb7); 
  basic_block(model.l4_b1_conv1, model.l4_b1_conv2, bb7, bb8); 

  Tensor avg;
  avgpool(bb8, model.pool, avg);

  print_shape(avg);

  Tensor fc_out;
  linear(avg, model.fc, fc_out);


  for (int i = 0; i<256; i+=1){
    cout << fc_out.data[i] << " ";
  }
  cout << endl;

}
