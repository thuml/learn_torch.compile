
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (28672L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x2 + (128L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x2 + (128L*x1) + (28672L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14400L + x2 + (128L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                        auto tmp4 = at::vec::maximum(tmp3, tmp1);
                        auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
                        auto tmp7 = at::vec::maximum(tmp6, tmp4);
                        auto tmp9 = at::vec::clamp_min(tmp8, decltype(tmp8)(0));
                        auto tmp10 = at::vec::maximum(tmp9, tmp7);
                        tmp10.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (7168L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_max_pool2d_with_indices_relu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (28672L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x2 + (256L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x2 + (256L*x1) + (28672L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14464L + x2 + (256L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                        auto tmp4 = at::vec::maximum(tmp3, tmp1);
                        auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
                        auto tmp7 = at::vec::maximum(tmp6, tmp4);
                        auto tmp9 = at::vec::clamp_min(tmp8, decltype(tmp8)(0));
                        auto tmp10 = at::vec::maximum(tmp9, tmp7);
                        tmp10.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (7168L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_relu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_max_pool2d_with_indices_relu_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (28672L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14592L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                        auto tmp4 = at::vec::maximum(tmp3, tmp1);
                        auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
                        auto tmp7 = at::vec::maximum(tmp6, tmp4);
                        auto tmp9 = at::vec::clamp_min(tmp8, decltype(tmp8)(0));
                        auto tmp10 = at::vec::maximum(tmp9, tmp7);
                        tmp10.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (7168L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_relu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_max_pool2d_with_indices_relu_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14848L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                        auto tmp4 = at::vec::maximum(tmp3, tmp1);
                        auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
                        auto tmp7 = at::vec::maximum(tmp6, tmp4);
                        auto tmp9 = at::vec::clamp_min(tmp8, decltype(tmp8)(0));
                        auto tmp10 = at::vec::maximum(tmp9, tmp7);
                        tmp10.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (7168L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__adaptive_avg_pool2d_max_pool2d_with_indices_relu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7168L + x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7680L + x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                    auto tmp4 = at::vec::maximum(tmp3, tmp1);
                    auto tmp6 = at::vec::clamp_min(tmp5, decltype(tmp5)(0));
                    auto tmp7 = at::vec::maximum(tmp6, tmp4);
                    auto tmp9 = at::vec::clamp_min(tmp8, decltype(tmp8)(0));
                    auto tmp10 = at::vec::maximum(tmp9, tmp7);
                    tmp10.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (3584L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (25088L*x0)), static_cast<long>(512L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (25088L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_relu_14 = async_compile.cpp('''
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


cpp_fused_relu_15 = async_compile.cpp('''
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (4096, 25088), (25088, 1))
    assert_size_stride(arg27_1, (4096, ), (1, ))
    assert_size_stride(arg28_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg29_1, (4096, ), (1, ))
    assert_size_stride(arg30_1, (1000, 4096), (4096, 1))
    assert_size_stride(arg31_1, (1000, ), (1, ))
    assert_size_stride(arg32_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg32_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg32_1
    # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg1_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (4, 64, 224, 224), (3211264, 1, 14336, 64))
    del arg1_1
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg2_1
    # Source Nodes: [l__mod___features_1, l__mod___features_2], Original ATen: [aten.convolution, aten.relu]
    buf5 = extern_kernels.convolution(buf3, buf4, arg3_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf5, (4, 64, 224, 224), (3211264, 1, 14336, 64))
    del arg3_1
    del buf3
    del buf4
    buf6 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_max_pool2d_with_indices_relu_2(c_void_p(buf5.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg4_1
    del buf5
    # Source Nodes: [l__mod___features_3, l__mod___features_4, l__mod___features_5], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
    buf8 = extern_kernels.convolution(buf6, buf7, arg5_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf8, (4, 128, 112, 112), (1605632, 1, 14336, 128))
    del arg5_1
    del buf6
    del buf7
    buf9 = buf8; del buf8  # reuse
    buf10 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_relu_3(c_void_p(buf9.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf10.data_ptr()))
    del arg6_1
    # Source Nodes: [l__mod___features_6, l__mod___features_7], Original ATen: [aten.convolution, aten.relu]
    buf11 = extern_kernels.convolution(buf9, buf10, arg7_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf11, (4, 128, 112, 112), (1605632, 1, 14336, 128))
    del arg7_1
    del buf10
    del buf9
    buf12 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_max_pool2d_with_indices_relu_4(c_void_p(buf11.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg8_1
    del buf11
    # Source Nodes: [l__mod___features_10, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
    buf14 = extern_kernels.convolution(buf12, buf13, arg9_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf14, (4, 256, 56, 56), (802816, 1, 14336, 256))
    del arg9_1
    del buf12
    del buf13
    buf15 = buf14; del buf14  # reuse
    buf16 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_relu_5(c_void_p(buf15.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(buf16.data_ptr()))
    del arg10_1
    # Source Nodes: [l__mod___features_11, l__mod___features_12], Original ATen: [aten.convolution, aten.relu]
    buf17 = extern_kernels.convolution(buf15, buf16, arg11_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf17, (4, 256, 56, 56), (802816, 1, 14336, 256))
    del arg11_1
    del buf15
    buf18 = buf17; del buf17  # reuse
    buf19 = buf16; del buf16  # reuse
    cpp_fused_convolution_relu_6(c_void_p(buf18.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf19.data_ptr()))
    del arg12_1
    # Source Nodes: [l__mod___features_13, l__mod___features_14], Original ATen: [aten.convolution, aten.relu]
    buf20 = extern_kernels.convolution(buf18, buf19, arg13_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf20, (4, 256, 56, 56), (802816, 1, 14336, 256))
    del arg13_1
    del buf18
    del buf19
    buf21 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_max_pool2d_with_indices_relu_7(c_void_p(buf20.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del arg14_1
    del buf20
    # Source Nodes: [l__mod___features_15, l__mod___features_16, l__mod___features_17], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
    buf23 = extern_kernels.convolution(buf21, buf22, arg15_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf23, (4, 512, 28, 28), (401408, 1, 14336, 512))
    del arg15_1
    del buf21
    del buf22
    buf24 = buf23; del buf23  # reuse
    buf25 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_relu_8(c_void_p(buf24.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf25.data_ptr()))
    del arg16_1
    # Source Nodes: [l__mod___features_18, l__mod___features_19], Original ATen: [aten.convolution, aten.relu]
    buf26 = extern_kernels.convolution(buf24, buf25, arg17_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf26, (4, 512, 28, 28), (401408, 1, 14336, 512))
    del arg17_1
    del buf24
    buf27 = buf26; del buf26  # reuse
    buf28 = buf25; del buf25  # reuse
    cpp_fused_convolution_relu_9(c_void_p(buf27.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(buf28.data_ptr()))
    del arg18_1
    # Source Nodes: [l__mod___features_20, l__mod___features_21], Original ATen: [aten.convolution, aten.relu]
    buf29 = extern_kernels.convolution(buf27, buf28, arg19_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf29, (4, 512, 28, 28), (401408, 1, 14336, 512))
    del arg19_1
    del buf27
    buf30 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    buf31 = buf28; del buf28  # reuse
    cpp_fused_convolution_max_pool2d_with_indices_relu_10(c_void_p(buf29.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg20_1
    del buf29
    # Source Nodes: [l__mod___features_22, l__mod___features_23, l__mod___features_24], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
    buf32 = extern_kernels.convolution(buf30, buf31, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf32, (4, 512, 14, 14), (100352, 1, 7168, 512))
    del arg21_1
    del buf30
    buf33 = buf32; del buf32  # reuse
    buf34 = buf31; del buf31  # reuse
    cpp_fused_convolution_relu_11(c_void_p(buf33.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf34.data_ptr()))
    del arg22_1
    # Source Nodes: [l__mod___features_25, l__mod___features_26], Original ATen: [aten.convolution, aten.relu]
    buf35 = extern_kernels.convolution(buf33, buf34, arg23_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf35, (4, 512, 14, 14), (100352, 1, 7168, 512))
    del arg23_1
    del buf33
    buf36 = buf35; del buf35  # reuse
    buf37 = buf34; del buf34  # reuse
    cpp_fused_convolution_relu_12(c_void_p(buf36.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf37.data_ptr()))
    del arg24_1
    # Source Nodes: [l__mod___features_27, l__mod___features_28], Original ATen: [aten.convolution, aten.relu]
    buf38 = extern_kernels.convolution(buf36, buf37, arg25_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf38, (4, 512, 14, 14), (100352, 1, 7168, 512))
    del arg25_1
    del buf36
    del buf37
    buf39 = empty_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    buf40 = empty((4, 512, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused__adaptive_avg_pool2d_max_pool2d_with_indices_relu_13(c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    del buf38
    del buf39
    buf41 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg27_1, reinterpret_tensor(buf40, (4, 25088), (25088, 1), 0), reinterpret_tensor(arg26_1, (25088, 4096), (1, 25088), 0), alpha=1, beta=1, out=buf41)
    del arg26_1
    del arg27_1
    del buf40
    buf42 = buf41; del buf41  # reuse
    cpp_fused_relu_14(c_void_p(buf42.data_ptr()))
    buf43 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_1, l__mod___classifier_3], Original ATen: [aten.addmm, aten.relu]
    extern_kernels.addmm(arg29_1, buf42, reinterpret_tensor(arg28_1, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf43)
    del arg28_1
    del arg29_1
    del buf42
    buf44 = buf43; del buf43  # reuse
    cpp_fused_relu_15(c_void_p(buf44.data_ptr()))
    buf45 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_4, x_3], Original ATen: [aten.addmm, aten.relu]
    extern_kernels.addmm(arg31_1, buf44, reinterpret_tensor(arg30_1, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf45)
    del arg30_1
    del arg31_1
    return (buf45, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((4096, 25088), (25088, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((1000, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vgg16', benchmark_compiled_module)
