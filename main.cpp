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
    const Tensor & bn_mean, const Tensor & bn_var, const Tensor & bn_alpha, const Tensor & bn_beta
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
    if (bias != nullptr) // conv bn relu fused
    for (int i = 0; i < dstC; ++i){
      for (int j = 0; j < N; ++j){
        float x = dst[i*N + j];
        if (bias != nullptr){
	  x+=bias[i];
	}
	if (&bn_mean != nullptr){
          x = (x-bn_mean.data[i]) / sqrt(bn_var.data[i]+eps); 
	  x = x * bn_alpha.data[i];
	  x = x + bn_beta.data[i];
	}
	dst[i*N+j]=x;
      }
    }
    src += srcC*srcH*srcW;
    dst += dstC*dstH*dstW;
  }
}

int main(){
  /*
  Tensor t1;
  read_ckpt("test.bin", t1);
  Tensor t2;
  read_ckpt("test_img.bin", t2);

  vector<float> buf;
  vector<float> dst;

  int padX=0,padY=0,strideX=1, strideY=1,padH=0,padW=0,group=1;
  int dilationX=1,dilationY=1,kernelX=t1.w,kernelY=t1.h;

  int dstH = (t2.h+padY+padH-( dilationY*(kernelY-1)+1 ) )  / strideY + 1; 
  int dstW = (t2.w+padX+padW-( dilationX*(kernelX-1)+1 ) )  / strideX + 1;
  int dstC=t1.b;
  cout << "dstH " << dstH << " dstW " << dstW << " dstC " << dstC <<  endl;
  int dst_size=dstC*t2.b*dstH*dstW;
  int buf_size=(t1.w+t2.w)*(t2.h+t2.h)*dstC;
  buf.reserve(buf_size);
  dst.reserve(dst_size);
  cout << "buf size " << buf_size << " dst size " << dst_size << endl; 

  conv_bn_relu(t2.data.data(), t2.b, t2.c, t2.h, t2.w,
      kernelY, kernelX, dilationY,dilationX,
      strideY, strideX,padY, padX,padH,padW,
      group,t1.data.data(),  nullptr, 
      dst.data(), dstC, buf.data(), 
      nullptr, nullptr, nullptr, nullptr);
  for (int i = 0; i<20; i++){
    cout << dst[i] << " ";
  }
  cout << endl;
  */
  read_net("r18.bin");
}
