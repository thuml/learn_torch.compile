
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


cpp_fused_max_pool2d_with_indices_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(55L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(55L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (14208L*x1) + (788544L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7104L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7168L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7232L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0)));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14208L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0)));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14272L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0)));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0)));
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
                            tmp25.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (3520L*x1) + (193600L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(193600L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12100L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp7 = tmp6 * (tmp6>0);
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp9 = tmp0 >= tmp3;
                    auto tmp10 = static_cast<long>(128);
                    auto tmp11 = tmp0 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-64L) + x1 + (64L*x0))];
                        auto tmp14 = tmp13 * (tmp13>0);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp16 = tmp4 ? tmp8 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused_relu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(193600L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_max_pool2d_with_indices_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12100L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp7 = tmp6 * (tmp6>0);
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp9 = tmp0 >= tmp3;
                    auto tmp10 = static_cast<long>(128);
                    auto tmp11 = tmp0 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-64L) + x1 + (64L*x0))];
                        auto tmp14 = tmp13 * (tmp13>0);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp16 = tmp4 ? tmp8 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp16;
                }
            }
        }
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (256L*x2) + (14080L*x1) + (387200L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(128L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(7040L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(7168L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(7296L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(14080L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(14208L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(14336L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0)));
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            auto tmp8 = at::vec::maximum(tmp7, tmp6);
                            auto tmp10 = at::vec::maximum(tmp9, tmp8);
                            auto tmp12 = at::vec::maximum(tmp11, tmp10);
                            auto tmp14 = at::vec::maximum(tmp13, tmp12);
                            auto tmp16 = at::vec::maximum(tmp15, tmp14);
                            tmp16.store(out_ptr1 + static_cast<long>(x3 + (128L*x2) + (3456L*x1) + (93312L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(93312L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2916L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(128);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp7 = tmp6 * (tmp6>0);
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp9 = tmp0 >= tmp3;
                    auto tmp10 = static_cast<long>(256);
                    auto tmp11 = tmp0 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-128L) + x1 + (128L*x0))];
                        auto tmp14 = tmp13 * (tmp13>0);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp16 = tmp4 ? tmp8 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(93312L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_max_pool2d_with_indices_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2916L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(128);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp7 = tmp6 * (tmp6>0);
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp9 = tmp0 >= tmp3;
                    auto tmp10 = static_cast<long>(256);
                    auto tmp11 = tmp0 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-128L) + x1 + (128L*x0))];
                        auto tmp14 = tmp13 * (tmp13>0);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp16 = tmp4 ? tmp8 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp16;
                }
            }
        }
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (512L*x2) + (13824L*x1) + (186624L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(512L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(6912L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(7168L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(7424L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(13824L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(14080L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(14336L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0)));
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            auto tmp8 = at::vec::maximum(tmp7, tmp6);
                            auto tmp10 = at::vec::maximum(tmp9, tmp8);
                            auto tmp12 = at::vec::maximum(tmp11, tmp10);
                            auto tmp14 = at::vec::maximum(tmp13, tmp12);
                            auto tmp16 = at::vec::maximum(tmp15, tmp14);
                            tmp16.store(out_ptr1 + static_cast<long>(x3 + (256L*x2) + (3328L*x1) + (43264L*x0)));
                        }
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (48L*x2) + (432L*x0)), static_cast<long>(48L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (48L*x2) + (432L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = tmp6 * (tmp6>0);
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp9 = tmp0 >= tmp3;
                    auto tmp10 = static_cast<long>(384);
                    auto tmp11 = tmp0 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp14 = tmp13 * (tmp13>0);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp16 = tmp4 ? tmp8 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (384L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (48L*x2) + (432L*x0)), static_cast<long>(48L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (48L*x2) + (432L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = tmp6 * (tmp6>0);
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp9 = tmp0 >= tmp3;
                    auto tmp10 = static_cast<long>(384);
                    auto tmp11 = tmp0 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp14 = tmp13 * (tmp13>0);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp16 = tmp4 ? tmp8 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (384L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused_relu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(43264L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_cat_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(256);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp7 = tmp6 * (tmp6>0);
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp9 = tmp0 >= tmp3;
                    auto tmp10 = static_cast<long>(512);
                    auto tmp11 = tmp0 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-256L) + x1 + (256L*x0))];
                        auto tmp14 = tmp13 * (tmp13>0);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp16 = tmp4 ? tmp8 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(43264L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_cat_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(256);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp7 = tmp6 * (tmp6>0);
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp9 = tmp0 >= tmp3;
                    auto tmp10 = static_cast<long>(512);
                    auto tmp11 = tmp0 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-256L) + x1 + (256L*x0))];
                        auto tmp14 = tmp13 * (tmp13>0);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp16 = tmp4 ? tmp8 : tmp15;
                    out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused_mean_relu_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1000L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(169L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1000L*x2) + (169000L*x0)));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1000L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4000L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(169.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (16, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg15_1, (32, ), (1, ))
    assert_size_stride(arg16_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg21_1, (32, ), (1, ))
    assert_size_stride(arg22_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg27_1, (48, ), (1, ))
    assert_size_stride(arg28_1, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg29_1, (192, ), (1, ))
    assert_size_stride(arg30_1, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg31_1, (192, ), (1, ))
    assert_size_stride(arg32_1, (48, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg33_1, (48, ), (1, ))
    assert_size_stride(arg34_1, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg35_1, (192, ), (1, ))
    assert_size_stride(arg36_1, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg37_1, (192, ), (1, ))
    assert_size_stride(arg38_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (1000, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg51_1, (1000, ), (1, ))
    assert_size_stride(arg52_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg52_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg52_1
    # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg1_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (4, 64, 111, 111), (788544, 1, 7104, 64))
    del arg1_1
    del buf0
    del buf1
    buf3 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.float32)
    cpp_fused_max_pool2d_with_indices_relu_1(c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()))
    del buf2
    # Source Nodes: [getattr_l__mod___features___3___squeeze], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf3, arg2_1, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf4, (4, 16, 55, 55), (48400, 1, 880, 16))
    del arg2_1
    del arg3_1
    del buf3
    buf5 = buf4; del buf4  # reuse
    cpp_fused_relu_2(c_void_p(buf5.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___3___expand1x1, x], Original ATen: [aten.convolution, aten.relu]
    buf6 = extern_kernels.convolution(buf5, arg4_1, arg5_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf6, (4, 64, 55, 55), (193600, 1, 3520, 64))
    del arg4_1
    del arg5_1
    buf7 = empty_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_3(c_void_p(arg6_1.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg6_1
    # Source Nodes: [getattr_l__mod___features___3___expand3x3], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf5, buf7, arg7_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf8, (4, 64, 55, 55), (193600, 1, 3520, 64))
    del arg7_1
    del buf5
    buf9 = empty_strided((4, 128, 55, 55), (387200, 1, 7040, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_4(c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    del buf6
    del buf8
    # Source Nodes: [cat_15, getattr_l__mod___features___4___squeeze], Original ATen: [aten.cat, aten.convolution]
    buf10 = extern_kernels.convolution(buf9, arg8_1, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf10, (4, 16, 55, 55), (48400, 1, 880, 16))
    del arg8_1
    del arg9_1
    buf11 = buf10; del buf10  # reuse
    cpp_fused_relu_5(c_void_p(buf11.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___4___expand1x1, x_1], Original ATen: [aten.convolution, aten.relu]
    buf12 = extern_kernels.convolution(buf11, arg10_1, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf12, (4, 64, 55, 55), (193600, 1, 3520, 64))
    del arg10_1
    del arg11_1
    buf13 = buf7; del buf7  # reuse
    cpp_fused_convolution_6(c_void_p(arg12_1.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg12_1
    # Source Nodes: [getattr_l__mod___features___4___expand3x3], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf11, buf13, arg13_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf14, (4, 64, 55, 55), (193600, 1, 3520, 64))
    del arg13_1
    del buf11
    del buf13
    buf15 = buf9; del buf9  # reuse
    buf16 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_max_pool2d_with_indices_7(c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()))
    del buf12
    del buf14
    del buf15
    # Source Nodes: [getattr_l__mod___features___6___squeeze], Original ATen: [aten.convolution]
    buf17 = extern_kernels.convolution(buf16, arg14_1, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf17, (4, 32, 27, 27), (23328, 1, 864, 32))
    del arg14_1
    del arg15_1
    del buf16
    buf18 = buf17; del buf17  # reuse
    cpp_fused_relu_8(c_void_p(buf18.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___6___expand1x1, x_2], Original ATen: [aten.convolution, aten.relu]
    buf19 = extern_kernels.convolution(buf18, arg16_1, arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf19, (4, 128, 27, 27), (93312, 1, 3456, 128))
    del arg16_1
    del arg17_1
    buf20 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_9(c_void_p(arg18_1.data_ptr()), c_void_p(buf20.data_ptr()))
    del arg18_1
    # Source Nodes: [getattr_l__mod___features___6___expand3x3], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(buf18, buf20, arg19_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf21, (4, 128, 27, 27), (93312, 1, 3456, 128))
    del arg19_1
    del buf18
    buf22 = empty_strided((4, 256, 27, 27), (186624, 1, 6912, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_10(c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del buf19
    del buf21
    # Source Nodes: [cat_13, getattr_l__mod___features___7___squeeze], Original ATen: [aten.cat, aten.convolution]
    buf23 = extern_kernels.convolution(buf22, arg20_1, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf23, (4, 32, 27, 27), (23328, 1, 864, 32))
    del arg20_1
    del arg21_1
    buf24 = buf23; del buf23  # reuse
    cpp_fused_relu_11(c_void_p(buf24.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___7___expand1x1, x_3], Original ATen: [aten.convolution, aten.relu]
    buf25 = extern_kernels.convolution(buf24, arg22_1, arg23_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf25, (4, 128, 27, 27), (93312, 1, 3456, 128))
    del arg22_1
    del arg23_1
    buf26 = buf20; del buf20  # reuse
    cpp_fused_convolution_12(c_void_p(arg24_1.data_ptr()), c_void_p(buf26.data_ptr()))
    del arg24_1
    # Source Nodes: [getattr_l__mod___features___7___expand3x3], Original ATen: [aten.convolution]
    buf27 = extern_kernels.convolution(buf24, buf26, arg25_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf27, (4, 128, 27, 27), (93312, 1, 3456, 128))
    del arg25_1
    del buf24
    del buf26
    buf28 = buf22; del buf22  # reuse
    buf29 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_max_pool2d_with_indices_13(c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    del buf25
    del buf27
    del buf28
    # Source Nodes: [getattr_l__mod___features___9___squeeze], Original ATen: [aten.convolution]
    buf30 = extern_kernels.convolution(buf29, arg26_1, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf30, (4, 48, 13, 13), (8112, 1, 624, 48))
    del arg26_1
    del arg27_1
    del buf29
    buf31 = buf30; del buf30  # reuse
    cpp_fused_relu_14(c_void_p(buf31.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___9___expand1x1, x_4], Original ATen: [aten.convolution, aten.relu]
    buf32 = extern_kernels.convolution(buf31, arg28_1, arg29_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf32, (4, 192, 13, 13), (32448, 1, 2496, 192))
    del arg28_1
    del arg29_1
    buf33 = empty_strided((192, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_15(c_void_p(arg30_1.data_ptr()), c_void_p(buf33.data_ptr()))
    del arg30_1
    # Source Nodes: [getattr_l__mod___features___9___expand3x3], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf31, buf33, arg31_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf34, (4, 192, 13, 13), (32448, 1, 2496, 192))
    del arg31_1
    del buf31
    buf35 = empty_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_16(c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    del buf32
    del buf34
    # Source Nodes: [cat_11, getattr_l__mod___features___10___squeeze], Original ATen: [aten.cat, aten.convolution]
    buf36 = extern_kernels.convolution(buf35, arg32_1, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf36, (4, 48, 13, 13), (8112, 1, 624, 48))
    del arg32_1
    del arg33_1
    buf37 = buf36; del buf36  # reuse
    cpp_fused_relu_17(c_void_p(buf37.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___10___expand1x1, x_5], Original ATen: [aten.convolution, aten.relu]
    buf38 = extern_kernels.convolution(buf37, arg34_1, arg35_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf38, (4, 192, 13, 13), (32448, 1, 2496, 192))
    del arg34_1
    del arg35_1
    buf39 = buf33; del buf33  # reuse
    cpp_fused_convolution_18(c_void_p(arg36_1.data_ptr()), c_void_p(buf39.data_ptr()))
    del arg36_1
    # Source Nodes: [getattr_l__mod___features___10___expand3x3], Original ATen: [aten.convolution]
    buf40 = extern_kernels.convolution(buf37, buf39, arg37_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf40, (4, 192, 13, 13), (32448, 1, 2496, 192))
    del arg37_1
    del buf37
    del buf39
    buf41 = buf35; del buf35  # reuse
    cpp_fused_cat_19(c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    del buf38
    del buf40
    # Source Nodes: [cat_10, getattr_l__mod___features___11___squeeze], Original ATen: [aten.cat, aten.convolution]
    buf42 = extern_kernels.convolution(buf41, arg38_1, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf42, (4, 64, 13, 13), (10816, 1, 832, 64))
    del arg38_1
    del arg39_1
    del buf41
    buf43 = buf42; del buf42  # reuse
    cpp_fused_relu_20(c_void_p(buf43.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___11___expand1x1, x_6], Original ATen: [aten.convolution, aten.relu]
    buf44 = extern_kernels.convolution(buf43, arg40_1, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf44, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del arg40_1
    del arg41_1
    buf45 = empty_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_21(c_void_p(arg42_1.data_ptr()), c_void_p(buf45.data_ptr()))
    del arg42_1
    # Source Nodes: [getattr_l__mod___features___11___expand3x3], Original ATen: [aten.convolution]
    buf46 = extern_kernels.convolution(buf43, buf45, arg43_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf46, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del arg43_1
    del buf43
    buf47 = empty_strided((4, 512, 13, 13), (86528, 1, 6656, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_22(c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    del buf44
    del buf46
    # Source Nodes: [cat_9, getattr_l__mod___features___12___squeeze], Original ATen: [aten.cat, aten.convolution]
    buf48 = extern_kernels.convolution(buf47, arg44_1, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf48, (4, 64, 13, 13), (10816, 1, 832, 64))
    del arg44_1
    del arg45_1
    buf49 = buf48; del buf48  # reuse
    cpp_fused_relu_23(c_void_p(buf49.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___12___expand1x1, x_7], Original ATen: [aten.convolution, aten.relu]
    buf50 = extern_kernels.convolution(buf49, arg46_1, arg47_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf50, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del arg46_1
    del arg47_1
    buf51 = buf45; del buf45  # reuse
    cpp_fused_convolution_24(c_void_p(arg48_1.data_ptr()), c_void_p(buf51.data_ptr()))
    del arg48_1
    # Source Nodes: [getattr_l__mod___features___12___expand3x3], Original ATen: [aten.convolution]
    buf52 = extern_kernels.convolution(buf49, buf51, arg49_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf52, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del arg49_1
    del buf49
    del buf51
    buf53 = buf47; del buf47  # reuse
    cpp_fused_cat_25(c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    del buf50
    del buf52
    # Source Nodes: [cat_8, l__mod___classifier_1], Original ATen: [aten.cat, aten.convolution]
    buf54 = extern_kernels.convolution(buf53, arg50_1, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf54, (4, 1000, 13, 13), (169000, 1, 13000, 1000))
    del arg50_1
    del arg51_1
    del buf53
    buf55 = empty_strided((4, 1000, 1, 1), (1000, 1, 4000, 4000), device='cpu', dtype=torch.float32)
    buf56 = reinterpret_tensor(buf55, (4, 1000), (1000, 1), 0); del buf55  # reuse
    cpp_fused_mean_relu_view_26(c_void_p(buf56.data_ptr()), c_void_p(buf54.data_ptr()))
    return (buf56, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((16, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((48, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((1000, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('squeezenet1_1', benchmark_compiled_module)
