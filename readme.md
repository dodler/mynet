Only inference of resnet18, written only in c++.

inference on laptop with battery powersave performance. 
input image is 64x64 .

original (torch) :22 ms. 

v1 (very naive): 1600 ms. 
v2 (better gemm): 942 ms. 
v3 (-O3 compilation): 120 ms. 
