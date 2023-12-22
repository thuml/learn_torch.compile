
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


cpp_fused_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
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
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                }
            }
        }
    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (48L*x2) + (432L*x0)), static_cast<long>(48L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (48L*x2) + (432L*x0)));
                }
            }
        }
    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (48L*x2) + (432L*x0)), static_cast<long>(48L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr6 + static_cast<long>(x1 + (48L*x2) + (432L*x0)));
                }
            }
        }
    }
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr8[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
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
                        auto tmp0 = in_ptr9[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr9[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_max_pool2d_with_indices_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       long* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3154176L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(55L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(55L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (14208L*x1) + (788544L*x0))];
                            auto tmp1 = in_out_ptr0[static_cast<long>(64L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0))];
                            auto tmp3 = in_out_ptr0[static_cast<long>(128L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(7104L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0))];
                            auto tmp7 = in_out_ptr0[static_cast<long>(7168L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0))];
                            auto tmp9 = in_out_ptr0[static_cast<long>(7232L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0))];
                            auto tmp11 = in_out_ptr0[static_cast<long>(14208L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(14272L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0))];
                            auto tmp15 = in_out_ptr0[static_cast<long>(14336L + x3 + (128L*x2) + (14208L*x1) + (788544L*x0))];
                            auto tmp2 = max_propagate_nan(tmp1, tmp0);
                            auto tmp4 = max_propagate_nan(tmp3, tmp2);
                            auto tmp6 = max_propagate_nan(tmp5, tmp4);
                            auto tmp8 = max_propagate_nan(tmp7, tmp6);
                            auto tmp10 = max_propagate_nan(tmp9, tmp8);
                            auto tmp12 = max_propagate_nan(tmp11, tmp10);
                            auto tmp14 = max_propagate_nan(tmp13, tmp12);
                            auto tmp16 = max_propagate_nan(tmp15, tmp14);
                            auto tmp17 = tmp1 > tmp0;
                            auto tmp18 = c10::convert<long>(1L + (2L*x2) + (222L*x1));
                            auto tmp19 = c10::convert<long>((2L*x2) + (222L*x1));
                            auto tmp20 = tmp17 ? tmp18 : tmp19;
                            auto tmp21 = tmp3 > tmp2;
                            auto tmp22 = c10::convert<long>(2L + (2L*x2) + (222L*x1));
                            auto tmp23 = tmp21 ? tmp22 : tmp20;
                            auto tmp24 = tmp5 > tmp4;
                            auto tmp25 = c10::convert<long>(111L + (2L*x2) + (222L*x1));
                            auto tmp26 = tmp24 ? tmp25 : tmp23;
                            auto tmp27 = tmp7 > tmp6;
                            auto tmp28 = c10::convert<long>(112L + (2L*x2) + (222L*x1));
                            auto tmp29 = tmp27 ? tmp28 : tmp26;
                            auto tmp30 = tmp9 > tmp8;
                            auto tmp31 = c10::convert<long>(113L + (2L*x2) + (222L*x1));
                            auto tmp32 = tmp30 ? tmp31 : tmp29;
                            auto tmp33 = tmp11 > tmp10;
                            auto tmp34 = c10::convert<long>(222L + (2L*x2) + (222L*x1));
                            auto tmp35 = tmp33 ? tmp34 : tmp32;
                            auto tmp36 = tmp13 > tmp12;
                            auto tmp37 = c10::convert<long>(223L + (2L*x2) + (222L*x1));
                            auto tmp38 = tmp36 ? tmp37 : tmp35;
                            auto tmp39 = tmp15 > tmp14;
                            auto tmp40 = c10::convert<long>(224L + (2L*x2) + (222L*x1));
                            auto tmp41 = tmp39 ? tmp40 : tmp38;
                            out_ptr0[static_cast<long>(x3 + (64L*x2) + (3520L*x1) + (193600L*x0))] = tmp16;
                            out_ptr1[static_cast<long>(x3 + (64L*x2) + (3520L*x1) + (193600L*x0))] = tmp41;
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


cpp_fused_cat_3 = async_compile.cpp('''
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


cpp_fused_relu_4 = async_compile.cpp('''
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


cpp_fused_cat_max_pool2d_with_indices_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2)
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
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(x3 + (256L*x2) + (14080L*x1) + (387200L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(128L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(256L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0))];
                            auto tmp5 = out_ptr0[static_cast<long>(7040L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(7168L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(7296L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0))];
                            auto tmp11 = out_ptr0[static_cast<long>(14080L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0))];
                            auto tmp13 = out_ptr0[static_cast<long>(14208L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0))];
                            auto tmp15 = out_ptr0[static_cast<long>(14336L + x3 + (256L*x2) + (14080L*x1) + (387200L*x0))];
                            auto tmp2 = max_propagate_nan(tmp1, tmp0);
                            auto tmp4 = max_propagate_nan(tmp3, tmp2);
                            auto tmp6 = max_propagate_nan(tmp5, tmp4);
                            auto tmp8 = max_propagate_nan(tmp7, tmp6);
                            auto tmp10 = max_propagate_nan(tmp9, tmp8);
                            auto tmp12 = max_propagate_nan(tmp11, tmp10);
                            auto tmp14 = max_propagate_nan(tmp13, tmp12);
                            auto tmp16 = max_propagate_nan(tmp15, tmp14);
                            auto tmp17 = tmp1 > tmp0;
                            auto tmp18 = c10::convert<long>(1L + (2L*x2) + (110L*x1));
                            auto tmp19 = c10::convert<long>((2L*x2) + (110L*x1));
                            auto tmp20 = tmp17 ? tmp18 : tmp19;
                            auto tmp21 = tmp3 > tmp2;
                            auto tmp22 = c10::convert<long>(2L + (2L*x2) + (110L*x1));
                            auto tmp23 = tmp21 ? tmp22 : tmp20;
                            auto tmp24 = tmp5 > tmp4;
                            auto tmp25 = c10::convert<long>(55L + (2L*x2) + (110L*x1));
                            auto tmp26 = tmp24 ? tmp25 : tmp23;
                            auto tmp27 = tmp7 > tmp6;
                            auto tmp28 = c10::convert<long>(56L + (2L*x2) + (110L*x1));
                            auto tmp29 = tmp27 ? tmp28 : tmp26;
                            auto tmp30 = tmp9 > tmp8;
                            auto tmp31 = c10::convert<long>(57L + (2L*x2) + (110L*x1));
                            auto tmp32 = tmp30 ? tmp31 : tmp29;
                            auto tmp33 = tmp11 > tmp10;
                            auto tmp34 = c10::convert<long>(110L + (2L*x2) + (110L*x1));
                            auto tmp35 = tmp33 ? tmp34 : tmp32;
                            auto tmp36 = tmp13 > tmp12;
                            auto tmp37 = c10::convert<long>(111L + (2L*x2) + (110L*x1));
                            auto tmp38 = tmp36 ? tmp37 : tmp35;
                            auto tmp39 = tmp15 > tmp14;
                            auto tmp40 = c10::convert<long>(112L + (2L*x2) + (110L*x1));
                            auto tmp41 = tmp39 ? tmp40 : tmp38;
                            out_ptr1[static_cast<long>(x3 + (128L*x2) + (3456L*x1) + (93312L*x0))] = tmp16;
                            out_ptr2[static_cast<long>(x3 + (128L*x2) + (3456L*x1) + (93312L*x0))] = tmp41;
                        }
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(93312L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_cat_7 = async_compile.cpp('''
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


cpp_fused_cat_max_pool2d_with_indices_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2)
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
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(x3 + (512L*x2) + (13824L*x1) + (186624L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(256L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(512L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0))];
                            auto tmp5 = out_ptr0[static_cast<long>(6912L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(7168L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(7424L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0))];
                            auto tmp11 = out_ptr0[static_cast<long>(13824L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0))];
                            auto tmp13 = out_ptr0[static_cast<long>(14080L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0))];
                            auto tmp15 = out_ptr0[static_cast<long>(14336L + x3 + (512L*x2) + (13824L*x1) + (186624L*x0))];
                            auto tmp2 = max_propagate_nan(tmp1, tmp0);
                            auto tmp4 = max_propagate_nan(tmp3, tmp2);
                            auto tmp6 = max_propagate_nan(tmp5, tmp4);
                            auto tmp8 = max_propagate_nan(tmp7, tmp6);
                            auto tmp10 = max_propagate_nan(tmp9, tmp8);
                            auto tmp12 = max_propagate_nan(tmp11, tmp10);
                            auto tmp14 = max_propagate_nan(tmp13, tmp12);
                            auto tmp16 = max_propagate_nan(tmp15, tmp14);
                            auto tmp17 = tmp1 > tmp0;
                            auto tmp18 = c10::convert<long>(1L + (2L*x2) + (54L*x1));
                            auto tmp19 = c10::convert<long>((2L*x2) + (54L*x1));
                            auto tmp20 = tmp17 ? tmp18 : tmp19;
                            auto tmp21 = tmp3 > tmp2;
                            auto tmp22 = c10::convert<long>(2L + (2L*x2) + (54L*x1));
                            auto tmp23 = tmp21 ? tmp22 : tmp20;
                            auto tmp24 = tmp5 > tmp4;
                            auto tmp25 = c10::convert<long>(27L + (2L*x2) + (54L*x1));
                            auto tmp26 = tmp24 ? tmp25 : tmp23;
                            auto tmp27 = tmp7 > tmp6;
                            auto tmp28 = c10::convert<long>(28L + (2L*x2) + (54L*x1));
                            auto tmp29 = tmp27 ? tmp28 : tmp26;
                            auto tmp30 = tmp9 > tmp8;
                            auto tmp31 = c10::convert<long>(29L + (2L*x2) + (54L*x1));
                            auto tmp32 = tmp30 ? tmp31 : tmp29;
                            auto tmp33 = tmp11 > tmp10;
                            auto tmp34 = c10::convert<long>(54L + (2L*x2) + (54L*x1));
                            auto tmp35 = tmp33 ? tmp34 : tmp32;
                            auto tmp36 = tmp13 > tmp12;
                            auto tmp37 = c10::convert<long>(55L + (2L*x2) + (54L*x1));
                            auto tmp38 = tmp36 ? tmp37 : tmp35;
                            auto tmp39 = tmp15 > tmp14;
                            auto tmp40 = c10::convert<long>(56L + (2L*x2) + (54L*x1));
                            auto tmp41 = tmp39 ? tmp40 : tmp38;
                            out_ptr1[static_cast<long>(x3 + (256L*x2) + (3328L*x1) + (43264L*x0))] = tmp16;
                            out_ptr2[static_cast<long>(x3 + (256L*x2) + (3328L*x1) + (43264L*x0))] = tmp41;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_10 = async_compile.cpp('''
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


cpp_fused_cat_11 = async_compile.cpp('''
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


cpp_fused_relu_12 = async_compile.cpp('''
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


cpp_fused_cat_13 = async_compile.cpp('''
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


cpp_fused_relu_14 = async_compile.cpp('''
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


cpp_fused_cat_15 = async_compile.cpp('''
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


cpp_fused_relu_16 = async_compile.cpp('''
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


cpp_fused_cat_17 = async_compile.cpp('''
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


cpp_fused_mean_relu_threshold_backward_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       bool* out_ptr1,
                       bool* out_ptr2,
                       bool* out_ptr3,
                       bool* out_ptr4,
                       bool* out_ptr5,
                       bool* out_ptr6,
                       bool* out_ptr7,
                       bool* out_ptr8,
                       bool* out_ptr9,
                       bool* out_ptr10,
                       bool* out_ptr11,
                       bool* out_ptr12,
                       bool* out_ptr13,
                       bool* out_ptr14,
                       bool* out_ptr15,
                       bool* out_ptr16,
                       bool* out_ptr17)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(676000L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr1[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(173056L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr1[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr2[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(173056L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr2[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr3[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(173056L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr3[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr4[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(173056L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr4[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr5[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(129792L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr5[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr6[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(129792L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr6[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr7[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(129792L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr7[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr8[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(129792L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr8[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr9[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(373248L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr9[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr10[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(373248L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr10[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr11[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(373248L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr11[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr12[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(373248L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr12[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr13[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(774400L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr13[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr14[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(774400L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr14[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr15[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(774400L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr15[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr16[static_cast<long>(x0)] = tmp3;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(774400L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr16[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr17[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (16, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_22, (32, ), (1, ))
    assert_size_stride(primals_23, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_28, (48, ), (1, ))
    assert_size_stride(primals_29, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_30, (192, ), (1, ))
    assert_size_stride(primals_31, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (48, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_34, (48, ), (1, ))
    assert_size_stride(primals_35, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_36, (192, ), (1, ))
    assert_size_stride(primals_37, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_39, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_42, (256, ), (1, ))
    assert_size_stride(primals_43, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (1000, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_52, (1000, ), (1, ))
    assert_size_stride(primals_53, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((192, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((192, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    del primals_1
    del primals_13
    del primals_19
    del primals_25
    del primals_31
    del primals_37
    del primals_43
    del primals_49
    del primals_53
    del primals_7
    # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
    buf10 = extern_kernels.convolution(buf9, buf0, primals_2, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf10, (4, 64, 111, 111), (788544, 1, 7104, 64))
    del primals_2
    buf11 = buf10; del buf10  # reuse
    buf12 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.int64)
    cpp_fused_max_pool2d_with_indices_relu_1(c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___3___squeeze], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf12, primals_3, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf14, (4, 16, 55, 55), (48400, 1, 880, 16))
    del primals_4
    buf15 = buf14; del buf14  # reuse
    cpp_fused_relu_2(c_void_p(buf15.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___3___expand1x1], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf15, primals_5, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf16, (4, 64, 55, 55), (193600, 1, 3520, 64))
    del primals_6
    # Source Nodes: [getattr_l__mod___features___3___expand3x3], Original ATen: [aten.convolution]
    buf17 = extern_kernels.convolution(buf15, buf1, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf17, (4, 64, 55, 55), (193600, 1, 3520, 64))
    del primals_8
    buf18 = empty_strided((4, 128, 55, 55), (387200, 1, 7040, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_3(c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___4___squeeze], Original ATen: [aten.convolution]
    buf19 = extern_kernels.convolution(buf18, primals_9, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf19, (4, 16, 55, 55), (48400, 1, 880, 16))
    del primals_10
    buf20 = buf19; del buf19  # reuse
    cpp_fused_relu_4(c_void_p(buf20.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___4___expand1x1], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(buf20, primals_11, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf21, (4, 64, 55, 55), (193600, 1, 3520, 64))
    del primals_12
    # Source Nodes: [getattr_l__mod___features___4___expand3x3], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(buf20, buf2, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf22, (4, 64, 55, 55), (193600, 1, 3520, 64))
    del primals_14
    buf23 = empty_strided((4, 128, 55, 55), (387200, 1, 7040, 128), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.int64)
    cpp_fused_cat_max_pool2d_with_indices_5(c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___6___squeeze], Original ATen: [aten.convolution]
    buf26 = extern_kernels.convolution(buf24, primals_15, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf26, (4, 32, 27, 27), (23328, 1, 864, 32))
    del primals_16
    buf27 = buf26; del buf26  # reuse
    cpp_fused_relu_6(c_void_p(buf27.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___6___expand1x1], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(buf27, primals_17, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf28, (4, 128, 27, 27), (93312, 1, 3456, 128))
    del primals_18
    # Source Nodes: [getattr_l__mod___features___6___expand3x3], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(buf27, buf3, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf29, (4, 128, 27, 27), (93312, 1, 3456, 128))
    del primals_20
    buf30 = empty_strided((4, 256, 27, 27), (186624, 1, 6912, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_7(c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___7___squeeze], Original ATen: [aten.convolution]
    buf31 = extern_kernels.convolution(buf30, primals_21, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf31, (4, 32, 27, 27), (23328, 1, 864, 32))
    del primals_22
    buf32 = buf31; del buf31  # reuse
    cpp_fused_relu_8(c_void_p(buf32.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___7___expand1x1], Original ATen: [aten.convolution]
    buf33 = extern_kernels.convolution(buf32, primals_23, primals_24, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf33, (4, 128, 27, 27), (93312, 1, 3456, 128))
    del primals_24
    # Source Nodes: [getattr_l__mod___features___7___expand3x3], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf32, buf4, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf34, (4, 128, 27, 27), (93312, 1, 3456, 128))
    del primals_26
    buf35 = empty_strided((4, 256, 27, 27), (186624, 1, 6912, 256), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.float32)
    buf37 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.int64)
    cpp_fused_cat_max_pool2d_with_indices_9(c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___9___squeeze], Original ATen: [aten.convolution]
    buf38 = extern_kernels.convolution(buf36, primals_27, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf38, (4, 48, 13, 13), (8112, 1, 624, 48))
    del primals_28
    buf39 = buf38; del buf38  # reuse
    cpp_fused_relu_10(c_void_p(buf39.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___9___expand1x1], Original ATen: [aten.convolution]
    buf40 = extern_kernels.convolution(buf39, primals_29, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf40, (4, 192, 13, 13), (32448, 1, 2496, 192))
    del primals_30
    # Source Nodes: [getattr_l__mod___features___9___expand3x3], Original ATen: [aten.convolution]
    buf41 = extern_kernels.convolution(buf39, buf5, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf41, (4, 192, 13, 13), (32448, 1, 2496, 192))
    del primals_32
    buf42 = empty_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_11(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___10___squeeze], Original ATen: [aten.convolution]
    buf43 = extern_kernels.convolution(buf42, primals_33, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf43, (4, 48, 13, 13), (8112, 1, 624, 48))
    del primals_34
    buf44 = buf43; del buf43  # reuse
    cpp_fused_relu_12(c_void_p(buf44.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___10___expand1x1], Original ATen: [aten.convolution]
    buf45 = extern_kernels.convolution(buf44, primals_35, primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf45, (4, 192, 13, 13), (32448, 1, 2496, 192))
    del primals_36
    # Source Nodes: [getattr_l__mod___features___10___expand3x3], Original ATen: [aten.convolution]
    buf46 = extern_kernels.convolution(buf44, buf6, primals_38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf46, (4, 192, 13, 13), (32448, 1, 2496, 192))
    del primals_38
    buf47 = empty_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_13(c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___11___squeeze], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(buf47, primals_39, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf48, (4, 64, 13, 13), (10816, 1, 832, 64))
    del primals_40
    buf49 = buf48; del buf48  # reuse
    cpp_fused_relu_14(c_void_p(buf49.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___11___expand1x1], Original ATen: [aten.convolution]
    buf50 = extern_kernels.convolution(buf49, primals_41, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf50, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del primals_42
    # Source Nodes: [getattr_l__mod___features___11___expand3x3], Original ATen: [aten.convolution]
    buf51 = extern_kernels.convolution(buf49, buf7, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf51, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del primals_44
    buf52 = empty_strided((4, 512, 13, 13), (86528, 1, 6656, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_15(c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___12___squeeze], Original ATen: [aten.convolution]
    buf53 = extern_kernels.convolution(buf52, primals_45, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf53, (4, 64, 13, 13), (10816, 1, 832, 64))
    del primals_46
    buf54 = buf53; del buf53  # reuse
    cpp_fused_relu_16(c_void_p(buf54.data_ptr()))
    # Source Nodes: [getattr_l__mod___features___12___expand1x1], Original ATen: [aten.convolution]
    buf55 = extern_kernels.convolution(buf54, primals_47, primals_48, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf55, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del primals_48
    # Source Nodes: [getattr_l__mod___features___12___expand3x3], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf54, buf8, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf56, (4, 256, 13, 13), (43264, 1, 3328, 256))
    del primals_50
    buf57 = empty_strided((4, 512, 13, 13), (86528, 1, 6656, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_17(c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()))
    # Source Nodes: [l__mod___classifier_1], Original ATen: [aten.convolution]
    buf58 = extern_kernels.convolution(buf57, primals_51, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf58, (4, 1000, 13, 13), (169000, 1, 13000, 1000))
    del primals_52
    buf59 = empty_strided((4, 1000, 1, 1), (1000, 1, 4000, 4000), device='cpu', dtype=torch.float32)
    buf60 = reinterpret_tensor(buf59, (4, 1000), (1000, 1), 0); del buf59  # reuse
    buf61 = empty_strided((4, 1000, 13, 13), (169000, 1, 13000, 1000), device='cpu', dtype=torch.bool)
    buf62 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.bool)
    buf63 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.bool)
    buf64 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.bool)
    buf65 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.bool)
    buf66 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.bool)
    buf67 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.bool)
    buf68 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.bool)
    buf69 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.bool)
    buf70 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.bool)
    buf71 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.bool)
    buf72 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.bool)
    buf73 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cpu', dtype=torch.bool)
    buf74 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.bool)
    buf75 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.bool)
    buf76 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.bool)
    buf77 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.bool)
    cpp_fused_mean_relu_threshold_backward_view_18(c_void_p(buf60.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()))
    return (buf60, buf0, primals_3, primals_5, buf1, primals_9, primals_11, buf2, primals_15, primals_17, buf3, primals_21, primals_23, buf4, primals_27, primals_29, buf5, primals_33, primals_35, buf6, primals_39, primals_41, buf7, primals_45, primals_47, buf8, primals_51, buf9, buf11, buf12, buf13, buf15, buf18, buf20, buf23, buf24, buf25, buf27, buf30, buf32, buf35, buf36, buf37, buf39, buf42, buf44, buf47, buf49, buf52, buf54, buf57, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((16, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((48, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((1000, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('squeezenet1_1', benchmark_compiled_module)
