
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


cpp_fused_convolution_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(121L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (121L*x1) + (363L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (363L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_max_pool2d_with_indices_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(27L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(27L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (7040L*x1) + (193600L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3520L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3584L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3648L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0)));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7040L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0)));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7104L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0)));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7168L + x3 + (128L*x2) + (7040L*x1) + (193600L*x0)));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                            auto tmp4 = at::vec::maximum(tmp3, tmp1);
                            auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
                            auto tmp7 = at::vec::maximum(tmp6, tmp4);
                            auto tmp9 = at::vec::clamp_min(tmp8, decltype(tmp8)(0));
                            auto tmp10 = at::vec::maximum(tmp9, tmp7);
                            auto tmp12 = at::vec::clamp_min(tmp11, decltype(tmp11)(0));
                            auto tmp13 = at::vec::maximum(tmp12, tmp10);
                            auto tmp15 = at::vec::clamp_min(tmp14, decltype(tmp14)(0));
                            auto tmp16 = at::vec::maximum(tmp15, tmp13);
                            auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                            auto tmp19 = at::vec::maximum(tmp18, tmp16);
                            auto tmp21 = at::vec::clamp_min(tmp20, decltype(tmp20)(0));
                            auto tmp22 = at::vec::maximum(tmp21, tmp19);
                            auto tmp24 = at::vec::clamp_min(tmp23, decltype(tmp23)(0));
                            auto tmp25 = at::vec::maximum(tmp24, tmp22);
                            tmp25.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (1728L*x1) + (46656L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (25L*x1) + (25L*x1_inner) + (1600L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (64L*x2) + (1600L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(24L); x2<static_cast<long>(25L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (25L*x1) + (25L*x1_inner) + (1600L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (1600L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_max_pool2d_with_indices_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(13L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(13L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (10368L*x1) + (139968L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(5184L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(5376L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(5568L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0)));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(10368L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0)));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(10560L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0)));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(10752L + x3 + (384L*x2) + (10368L*x1) + (139968L*x0)));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                            auto tmp4 = at::vec::maximum(tmp3, tmp1);
                            auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
                            auto tmp7 = at::vec::maximum(tmp6, tmp4);
                            auto tmp9 = at::vec::clamp_min(tmp8, decltype(tmp8)(0));
                            auto tmp10 = at::vec::maximum(tmp9, tmp7);
                            auto tmp12 = at::vec::clamp_min(tmp11, decltype(tmp11)(0));
                            auto tmp13 = at::vec::maximum(tmp12, tmp10);
                            auto tmp15 = at::vec::clamp_min(tmp14, decltype(tmp14)(0));
                            auto tmp16 = at::vec::maximum(tmp15, tmp13);
                            auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                            auto tmp19 = at::vec::maximum(tmp18, tmp16);
                            auto tmp21 = at::vec::clamp_min(tmp20, decltype(tmp20)(0));
                            auto tmp22 = at::vec::maximum(tmp21, tmp19);
                            auto tmp24 = at::vec::clamp_min(tmp23, decltype(tmp23)(0));
                            auto tmp25 = at::vec::maximum(tmp24, tmp22);
                            tmp25.store(out_ptr0 + static_cast<long>(x3 + (192L*x2) + (2496L*x1) + (32448L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(259584L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_relu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(173056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__adaptive_avg_pool2d_max_pool2d_with_indices_relu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (6656L*x1) + (43264L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3328L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3584L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3840L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(6656L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(6912L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0)));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7168L + x3 + (512L*x2) + (6656L*x1) + (43264L*x0)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                        auto tmp4 = at::vec::maximum(tmp3, tmp1);
                        auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
                        auto tmp7 = at::vec::maximum(tmp6, tmp4);
                        auto tmp9 = at::vec::clamp_min(tmp8, decltype(tmp8)(0));
                        auto tmp10 = at::vec::maximum(tmp9, tmp7);
                        auto tmp12 = at::vec::clamp_min(tmp11, decltype(tmp11)(0));
                        auto tmp13 = at::vec::maximum(tmp12, tmp10);
                        auto tmp15 = at::vec::clamp_min(tmp14, decltype(tmp14)(0));
                        auto tmp16 = at::vec::maximum(tmp15, tmp13);
                        auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                        auto tmp19 = at::vec::maximum(tmp18, tmp16);
                        auto tmp21 = at::vec::clamp_min(tmp20, decltype(tmp20)(0));
                        auto tmp22 = at::vec::maximum(tmp21, tmp19);
                        auto tmp24 = at::vec::clamp_min(tmp23, decltype(tmp23)(0));
                        auto tmp25 = at::vec::maximum(tmp24, tmp22);
                        tmp25.store(out_ptr0 + static_cast<long>(x3 + (256L*x2) + (1536L*x1) + (9216L*x0)));
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (9216L*x0)), static_cast<long>(256L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        tmp1.store(out_ptr1 + static_cast<long>(x2 + (36L*x1) + (36L*x1_inner) + (9216L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(32L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (9216L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (36L*x1) + (36L*x1_inner) + (9216L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 11, 11), (363, 121, 11, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (192, 64, 5, 5), (1600, 25, 5, 1))
    assert_size_stride(arg3_1, (192, ), (1, ))
    assert_size_stride(arg4_1, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg5_1, (384, ), (1, ))
    assert_size_stride(arg6_1, (256, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg7_1, (256, ), (1, ))
    assert_size_stride(arg8_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (4096, 9216), (9216, 1))
    assert_size_stride(arg11_1, (4096, ), (1, ))
    assert_size_stride(arg12_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg13_1, (4096, ), (1, ))
    assert_size_stride(arg14_1, (1000, 4096), (4096, 1))
    assert_size_stride(arg15_1, (1000, ), (1, ))
    assert_size_stride(arg16_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 11, 11), (363, 1, 33, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg16_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg16_1
    # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg1_1, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (4, 64, 55, 55), (193600, 1, 3520, 64))
    del arg1_1
    del buf0
    del buf1
    buf3 = empty_strided((4, 64, 27, 27), (46656, 1, 1728, 64), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((192, 64, 5, 5), (1600, 1, 320, 64), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_max_pool2d_with_indices_relu_1(c_void_p(buf2.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg2_1
    del buf2
    # Source Nodes: [l__mod___features_3], Original ATen: [aten.convolution]
    buf5 = extern_kernels.convolution(buf3, buf4, arg3_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf5, (4, 192, 27, 27), (139968, 1, 5184, 192))
    del arg3_1
    del buf3
    del buf4
    buf6 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_max_pool2d_with_indices_relu_2(c_void_p(buf5.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg4_1
    del buf5
    # Source Nodes: [l__mod___features_6], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf6, buf7, arg5_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf8, (4, 384, 13, 13), (64896, 1, 4992, 384))
    del arg5_1
    del buf6
    del buf7
    buf9 = buf8; del buf8  # reuse
    buf10 = empty_strided((256, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_relu_3(c_void_p(buf9.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf10.data_ptr()))
    del arg6_1
    # Source Nodes: [l__mod___features_7, l__mod___features_8], Original ATen: [aten.convolution, aten.relu]
    buf11 = extern_kernels.convolution(buf9, buf10, arg7_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf11, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del arg7_1
    del buf10
    del buf9
    buf12 = buf11; del buf11  # reuse
    buf13 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_relu_4(c_void_p(buf12.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg8_1
    # Source Nodes: [l__mod___features_10, l__mod___features_9], Original ATen: [aten.convolution, aten.relu]
    buf14 = extern_kernels.convolution(buf12, buf13, arg9_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf14, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del arg9_1
    del buf12
    del buf13
    buf15 = empty_strided((4, 256, 6, 6), (9216, 1, 1536, 256), device='cpu', dtype=torch.float32)
    buf16 = empty((4, 256, 6, 6), device='cpu', dtype=torch.float32)
    cpp_fused__adaptive_avg_pool2d_max_pool2d_with_indices_relu_5(c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()))
    del buf14
    del buf15
    buf17 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg11_1, reinterpret_tensor(buf16, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg10_1, (9216, 4096), (1, 9216), 0), alpha=1, beta=1, out=buf17)
    del arg10_1
    del arg11_1
    del buf16
    buf18 = buf17; del buf17  # reuse
    cpp_fused_relu_6(c_void_p(buf18.data_ptr()))
    buf19 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_2, l__mod___classifier_4], Original ATen: [aten.addmm, aten.relu]
    extern_kernels.addmm(arg13_1, buf18, reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf19)
    del arg12_1
    del arg13_1
    del buf18
    buf20 = buf19; del buf19  # reuse
    cpp_fused_relu_7(c_void_p(buf20.data_ptr()))
    buf21 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_5, x_3], Original ATen: [aten.addmm, aten.relu]
    extern_kernels.addmm(arg15_1, buf20, reinterpret_tensor(arg14_1, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf21)
    del arg14_1
    del arg15_1
    return (buf21, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 11, 11), (363, 121, 11, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((192, 64, 5, 5), (1600, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((256, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((4096, 9216), (9216, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((1000, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('alexnet', benchmark_compiled_module)
