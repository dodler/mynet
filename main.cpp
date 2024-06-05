#include <iostream>
#include <vector>
#include "net_io.cpp"
#include <cmath>
#include "utils.h"

using namespace std;
float relu(float value){
  return value > 0 ? value : 0;
}

void im2col(const float * src, int srcC, int srcH, int srcW,
    int kernelY, int kernelX, int dilationY, int dilationX,
    int strideY, int strideX, int padY, int padX, int padH, int padW, float * buf)
{
  int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
  int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
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

void conv_bn_relu(const float * src, int batch, int srcC, int srcH, int srcW,
    int kernelY, int kernelX, int dilationY, int dilationX,
    int strideY, int strideX, int padY, int padX, int padH, int padW, int group,
    const float * weight, const float * bias, float * dst, int dstC, float * buf,
    const vector<float> & bn_mean, const vector<float> & bn_var, 
    const vector<float> & bn_alpha, const vector<float> & bn_beta,
    bool do_relu
    )
{
  int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
  int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
  int M = dstC / group;
  int N = dstH * dstW;
  int K = srcC * kernelY * kernelX / group;
  const float eps = 1e-5;
  for (int b = 0; b < batch; ++b)
  {
    im2col(src, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX,
      strideY, strideX, padY, padX, padH, padW, buf);
    for (int g = 0; g < group; ++g)
      gemm_nn(M, N, K, 1, weight + M * K * g, K, 0, buf + N * K * g, N, dst + M * N * g, N);

    for (int i = 0; i < dstC; ++i){
      float mean=bn_mean[i];
      float var = sqrt(bn_var[i]+eps);
      float alpha = bn_alpha[i];
      float beta = bn_beta[i];
      for (int j = 0; j < N; ++j){
        float x = dst[i*N + j];
        if (bias != nullptr){
	  x+=bias[i];
	}
	if (&bn_mean != nullptr){
          x = (x-mean) / var; 
	  x = x * alpha;
	  x = x + beta;
	}
	if (do_relu == 1){
	  x=relu(x);
	}
	dst[i*N+j]=x;
      }
    }
    src += srcC*srcH*srcW;
    dst += dstC*dstH*dstW;
  }
}

int main(){
  //Tensor t1;
  //read_ckpt("test.bin", t1);
  Tensor img;
  read_ckpt("test_img.bin", img);
  cout << "img " << endl;
  for (int i = 0;i<20;i++){
    cout << img.data[i] << " ";
  }
  cout << endl;

  ConvBnRelu layer1;
  read_net("r18.bin", layer1);

  vector<float> buf;
  vector<float> dst;

  int dstH = (img.h+layer1.padY+layer1.padH-( layer1.dilationY*(layer1.kh-1)+1 ) )  / layer1.strideY + 1; 
  int dstW = (img.w+layer1.padX+layer1.padW-( layer1.dilationX*(layer1.kw-1)+1 ) )  / layer1.strideX + 1;
  int dstC=layer1.kd;
  cout << "dstH " << dstH << " dstW " << dstW << " dstC " << dstC <<  endl;
  int dst_size=dstC*img.b*dstH*dstW;
  int buf_size=(layer1.kw+img.w)*(layer1.kh+img.h)*dstC;
  buf.reserve(buf_size);
  dst.reserve(dst_size);
  cout << "buf size " << buf_size << " dst size " << dst_size << endl; 
  cout << "layer do relu " << layer1.doRelu << endl;

  conv_bn_relu(img.data.data(), img.b, img.c, img.h, img.w,
		  layer1.kh, layer1.kw, layer1.dilationY, layer1.dilationX, 
		  layer1.strideY, layer1.strideX, 
		  layer1.padY, layer1.padX, layer1.padH, layer1.padW, 
		  layer1.group, layer1.convWeight.data(), layer1.biasWeight.data(),
		  dst.data(), dstC,  buf.data(), 
		  layer1.bnRunningMean, layer1.bnRunningVar,
		  layer1.bnWeight, layer1.bnBias, layer1.doRelu 
		  );
  for (int i = 0; i<256; i++){
    cout << dst[i] << " ";
  }
  cout << endl;

}
