
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
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
                       const float* in_ptr33,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15,
                       float* out_ptr16,
                       float* out_ptr17,
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       float* out_ptr22,
                       float* out_ptr23,
                       float* out_ptr24,
                       float* out_ptr25,
                       float* out_ptr26,
                       float* out_ptr27,
                       float* out_ptr28,
                       float* out_ptr29,
                       float* out_ptr30,
                       float* out_ptr31,
                       float* out_ptr32,
                       float* out_ptr33)
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr8[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr9 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr9[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr10 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr10[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr11 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr11[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr12 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr12[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr13 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr13[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr14 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr14[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr15 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr15[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr16 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr16[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr17 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr17[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (6912L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr18 + static_cast<long>(x1 + (768L*x2) + (6912L*x0)), static_cast<long>(768L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr18[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (6912L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (768L*x2) + (6912L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr19 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr19[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr20 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr20[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr21 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr21[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr22 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr22[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (6912L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr23 + static_cast<long>(x1 + (768L*x2) + (6912L*x0)), static_cast<long>(768L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr23[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (6912L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (768L*x2) + (6912L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr24 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)), static_cast<long>(224L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr24[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr25 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)), static_cast<long>(224L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr25[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr26 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)), static_cast<long>(224L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr26[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr27 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)), static_cast<long>(224L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr27[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr28 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (9216L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr28 + static_cast<long>(x1 + (1024L*x2) + (9216L*x0)), static_cast<long>(1024L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr28[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (9216L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr28 + static_cast<long>(x1 + (1024L*x2) + (9216L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr29 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)), static_cast<long>(224L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr29[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr29 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr30 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr30 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)), static_cast<long>(224L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr30[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr30 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr31 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr31 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)), static_cast<long>(224L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr31[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr31 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr32 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr32 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)), static_cast<long>(224L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr32[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2016L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr32 + static_cast<long>(x1 + (224L*x2) + (2016L*x0)));
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
                        auto tmp0 = in_ptr33[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr33[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp15 <= tmp16;
                    out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp15;
                    out_ptr1[static_cast<long>(x1 + (128L*x0))] = tmp17;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(2L*x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>(2L*x2);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>(x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : -std::numeric_limits<decltype(tmp11())>::infinity();
                            auto tmp14 = c10::convert<long>(1L + (2L*x2));
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = out_ptr0[static_cast<long>(256L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : -std::numeric_limits<decltype(tmp19())>::infinity();
                            auto tmp22 = max_propagate_nan(tmp21, tmp13);
                            auto tmp23 = c10::convert<long>(2L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = out_ptr0[static_cast<long>(512L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : -std::numeric_limits<decltype(tmp28())>::infinity();
                            auto tmp31 = max_propagate_nan(tmp30, tmp22);
                            auto tmp32 = c10::convert<long>(1L + (2L*x1));
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = out_ptr0[static_cast<long>(14336L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : -std::numeric_limits<decltype(tmp37())>::infinity();
                            auto tmp40 = max_propagate_nan(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = out_ptr0[static_cast<long>(14592L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = max_propagate_nan(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = out_ptr0[static_cast<long>(14848L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : -std::numeric_limits<decltype(tmp47())>::infinity();
                            auto tmp50 = max_propagate_nan(tmp49, tmp45);
                            auto tmp51 = c10::convert<long>(2L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = out_ptr0[static_cast<long>(28672L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : -std::numeric_limits<decltype(tmp56())>::infinity();
                            auto tmp59 = max_propagate_nan(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = out_ptr0[static_cast<long>(28928L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : -std::numeric_limits<decltype(tmp61())>::infinity();
                            auto tmp64 = max_propagate_nan(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = out_ptr0[static_cast<long>(29184L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : -std::numeric_limits<decltype(tmp66())>::infinity();
                            auto tmp69 = max_propagate_nan(tmp68, tmp64);
                            auto tmp70 = [&]
                            {
                                auto tmp71 = out_ptr0[static_cast<long>(x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp10 ? tmp70() : -std::numeric_limits<decltype(tmp70())>::infinity();
                            auto tmp73 = [&]
                            {
                                auto tmp74 = out_ptr0[static_cast<long>(256L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp74;
                            }
                            ;
                            auto tmp75 = tmp18 ? tmp73() : -std::numeric_limits<decltype(tmp73())>::infinity();
                            auto tmp76 = tmp75 > tmp72;
                            auto tmp77 = c10::convert<long>(1L + (2L*x2) + (112L*x1));
                            auto tmp78 = c10::convert<long>((2L*x2) + (112L*x1));
                            auto tmp79 = tmp76 ? tmp77 : tmp78;
                            auto tmp80 = max_propagate_nan(tmp75, tmp72);
                            auto tmp81 = [&]
                            {
                                auto tmp82 = out_ptr0[static_cast<long>(512L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp82;
                            }
                            ;
                            auto tmp83 = tmp27 ? tmp81() : -std::numeric_limits<decltype(tmp81())>::infinity();
                            auto tmp84 = tmp83 > tmp80;
                            auto tmp85 = c10::convert<long>(2L + (2L*x2) + (112L*x1));
                            auto tmp86 = tmp84 ? tmp85 : tmp79;
                            auto tmp87 = max_propagate_nan(tmp83, tmp80);
                            auto tmp88 = [&]
                            {
                                auto tmp89 = out_ptr0[static_cast<long>(14336L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp89;
                            }
                            ;
                            auto tmp90 = tmp36 ? tmp88() : -std::numeric_limits<decltype(tmp88())>::infinity();
                            auto tmp91 = tmp90 > tmp87;
                            auto tmp92 = c10::convert<long>(56L + (2L*x2) + (112L*x1));
                            auto tmp93 = tmp91 ? tmp92 : tmp86;
                            auto tmp94 = max_propagate_nan(tmp90, tmp87);
                            auto tmp95 = [&]
                            {
                                auto tmp96 = out_ptr0[static_cast<long>(14592L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp96;
                            }
                            ;
                            auto tmp97 = tmp41 ? tmp95() : -std::numeric_limits<decltype(tmp95())>::infinity();
                            auto tmp98 = tmp97 > tmp94;
                            auto tmp99 = c10::convert<long>(57L + (2L*x2) + (112L*x1));
                            auto tmp100 = tmp98 ? tmp99 : tmp93;
                            auto tmp101 = max_propagate_nan(tmp97, tmp94);
                            auto tmp102 = [&]
                            {
                                auto tmp103 = out_ptr0[static_cast<long>(14848L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp46 ? tmp102() : -std::numeric_limits<decltype(tmp102())>::infinity();
                            auto tmp105 = tmp104 > tmp101;
                            auto tmp106 = c10::convert<long>(58L + (2L*x2) + (112L*x1));
                            auto tmp107 = tmp105 ? tmp106 : tmp100;
                            auto tmp108 = max_propagate_nan(tmp104, tmp101);
                            auto tmp109 = [&]
                            {
                                auto tmp110 = out_ptr0[static_cast<long>(28672L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp110;
                            }
                            ;
                            auto tmp111 = tmp55 ? tmp109() : -std::numeric_limits<decltype(tmp109())>::infinity();
                            auto tmp112 = tmp111 > tmp108;
                            auto tmp113 = c10::convert<long>(112L + (2L*x2) + (112L*x1));
                            auto tmp114 = tmp112 ? tmp113 : tmp107;
                            auto tmp115 = max_propagate_nan(tmp111, tmp108);
                            auto tmp116 = [&]
                            {
                                auto tmp117 = out_ptr0[static_cast<long>(28928L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp117;
                            }
                            ;
                            auto tmp118 = tmp60 ? tmp116() : -std::numeric_limits<decltype(tmp116())>::infinity();
                            auto tmp119 = tmp118 > tmp115;
                            auto tmp120 = c10::convert<long>(113L + (2L*x2) + (112L*x1));
                            auto tmp121 = tmp119 ? tmp120 : tmp114;
                            auto tmp122 = max_propagate_nan(tmp118, tmp115);
                            auto tmp123 = [&]
                            {
                                auto tmp124 = out_ptr0[static_cast<long>(29184L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                                return tmp124;
                            }
                            ;
                            auto tmp125 = tmp65 ? tmp123() : -std::numeric_limits<decltype(tmp123())>::infinity();
                            auto tmp126 = tmp125 > tmp122;
                            auto tmp127 = c10::convert<long>(114L + (2L*x2) + (112L*x1));
                            auto tmp128 = tmp126 ? tmp127 : tmp121;
                            auto tmp129 = max_propagate_nan(tmp125, tmp122);
                            out_ptr1[static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0))] = tmp69;
                            out_ptr2[static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0))] = tmp128;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1056L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1056L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1056L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1056L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       bool* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (160L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp15 <= tmp16;
                    out_ptr0[static_cast<long>(x1 + (1056L*x0))] = tmp15;
                    out_ptr1[static_cast<long>(x1 + (160L*x0))] = tmp17;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x2) + (200704L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (1056L*x2) + (827904L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(2L*x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(28);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>(2L*x2);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>(x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : -std::numeric_limits<decltype(tmp11())>::infinity();
                            auto tmp14 = c10::convert<long>(1L + (2L*x2));
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = out_ptr0[static_cast<long>(512L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : -std::numeric_limits<decltype(tmp19())>::infinity();
                            auto tmp22 = max_propagate_nan(tmp21, tmp13);
                            auto tmp23 = c10::convert<long>(2L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = out_ptr0[static_cast<long>(1024L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : -std::numeric_limits<decltype(tmp28())>::infinity();
                            auto tmp31 = max_propagate_nan(tmp30, tmp22);
                            auto tmp32 = c10::convert<long>(1L + (2L*x1));
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = out_ptr0[static_cast<long>(14336L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : -std::numeric_limits<decltype(tmp37())>::infinity();
                            auto tmp40 = max_propagate_nan(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = out_ptr0[static_cast<long>(14848L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = max_propagate_nan(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = out_ptr0[static_cast<long>(15360L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : -std::numeric_limits<decltype(tmp47())>::infinity();
                            auto tmp50 = max_propagate_nan(tmp49, tmp45);
                            auto tmp51 = c10::convert<long>(2L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = out_ptr0[static_cast<long>(28672L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : -std::numeric_limits<decltype(tmp56())>::infinity();
                            auto tmp59 = max_propagate_nan(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = out_ptr0[static_cast<long>(29184L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : -std::numeric_limits<decltype(tmp61())>::infinity();
                            auto tmp64 = max_propagate_nan(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = out_ptr0[static_cast<long>(29696L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : -std::numeric_limits<decltype(tmp66())>::infinity();
                            auto tmp69 = max_propagate_nan(tmp68, tmp64);
                            auto tmp70 = [&]
                            {
                                auto tmp71 = out_ptr0[static_cast<long>(x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp10 ? tmp70() : -std::numeric_limits<decltype(tmp70())>::infinity();
                            auto tmp73 = [&]
                            {
                                auto tmp74 = out_ptr0[static_cast<long>(512L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp74;
                            }
                            ;
                            auto tmp75 = tmp18 ? tmp73() : -std::numeric_limits<decltype(tmp73())>::infinity();
                            auto tmp76 = tmp75 > tmp72;
                            auto tmp77 = c10::convert<long>(1L + (2L*x2) + (56L*x1));
                            auto tmp78 = c10::convert<long>((2L*x2) + (56L*x1));
                            auto tmp79 = tmp76 ? tmp77 : tmp78;
                            auto tmp80 = max_propagate_nan(tmp75, tmp72);
                            auto tmp81 = [&]
                            {
                                auto tmp82 = out_ptr0[static_cast<long>(1024L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp82;
                            }
                            ;
                            auto tmp83 = tmp27 ? tmp81() : -std::numeric_limits<decltype(tmp81())>::infinity();
                            auto tmp84 = tmp83 > tmp80;
                            auto tmp85 = c10::convert<long>(2L + (2L*x2) + (56L*x1));
                            auto tmp86 = tmp84 ? tmp85 : tmp79;
                            auto tmp87 = max_propagate_nan(tmp83, tmp80);
                            auto tmp88 = [&]
                            {
                                auto tmp89 = out_ptr0[static_cast<long>(14336L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp89;
                            }
                            ;
                            auto tmp90 = tmp36 ? tmp88() : -std::numeric_limits<decltype(tmp88())>::infinity();
                            auto tmp91 = tmp90 > tmp87;
                            auto tmp92 = c10::convert<long>(28L + (2L*x2) + (56L*x1));
                            auto tmp93 = tmp91 ? tmp92 : tmp86;
                            auto tmp94 = max_propagate_nan(tmp90, tmp87);
                            auto tmp95 = [&]
                            {
                                auto tmp96 = out_ptr0[static_cast<long>(14848L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp96;
                            }
                            ;
                            auto tmp97 = tmp41 ? tmp95() : -std::numeric_limits<decltype(tmp95())>::infinity();
                            auto tmp98 = tmp97 > tmp94;
                            auto tmp99 = c10::convert<long>(29L + (2L*x2) + (56L*x1));
                            auto tmp100 = tmp98 ? tmp99 : tmp93;
                            auto tmp101 = max_propagate_nan(tmp97, tmp94);
                            auto tmp102 = [&]
                            {
                                auto tmp103 = out_ptr0[static_cast<long>(15360L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp46 ? tmp102() : -std::numeric_limits<decltype(tmp102())>::infinity();
                            auto tmp105 = tmp104 > tmp101;
                            auto tmp106 = c10::convert<long>(30L + (2L*x2) + (56L*x1));
                            auto tmp107 = tmp105 ? tmp106 : tmp100;
                            auto tmp108 = max_propagate_nan(tmp104, tmp101);
                            auto tmp109 = [&]
                            {
                                auto tmp110 = out_ptr0[static_cast<long>(28672L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp110;
                            }
                            ;
                            auto tmp111 = tmp55 ? tmp109() : -std::numeric_limits<decltype(tmp109())>::infinity();
                            auto tmp112 = tmp111 > tmp108;
                            auto tmp113 = c10::convert<long>(56L + (2L*x2) + (56L*x1));
                            auto tmp114 = tmp112 ? tmp113 : tmp107;
                            auto tmp115 = max_propagate_nan(tmp111, tmp108);
                            auto tmp116 = [&]
                            {
                                auto tmp117 = out_ptr0[static_cast<long>(29184L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp117;
                            }
                            ;
                            auto tmp118 = tmp60 ? tmp116() : -std::numeric_limits<decltype(tmp116())>::infinity();
                            auto tmp119 = tmp118 > tmp115;
                            auto tmp120 = c10::convert<long>(57L + (2L*x2) + (56L*x1));
                            auto tmp121 = tmp119 ? tmp120 : tmp114;
                            auto tmp122 = max_propagate_nan(tmp118, tmp115);
                            auto tmp123 = [&]
                            {
                                auto tmp124 = out_ptr0[static_cast<long>(29696L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                                return tmp124;
                            }
                            ;
                            auto tmp125 = tmp65 ? tmp123() : -std::numeric_limits<decltype(tmp123())>::infinity();
                            auto tmp126 = tmp125 > tmp122;
                            auto tmp127 = c10::convert<long>(58L + (2L*x2) + (56L*x1));
                            auto tmp128 = tmp126 ? tmp127 : tmp121;
                            auto tmp129 = max_propagate_nan(tmp125, tmp122);
                            out_ptr1[static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0))] = tmp69;
                            out_ptr2[static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0))] = tmp128;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1472L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1472L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1472L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1472L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_threshold_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       bool* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp15 <= tmp16;
                    out_ptr0[static_cast<long>(x1 + (1472L*x0))] = tmp15;
                    out_ptr1[static_cast<long>(x1 + (192L*x0))] = tmp17;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (1472L*x2) + (288512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1728L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1728L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1728L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1728L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (1728L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = in_ptr3[static_cast<long>(x1)];
                    auto tmp13 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = std::sqrt(tmp5);
                    auto tmp7 = 1 / tmp6;
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = tmp14 * (tmp14>0);
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp15 <= tmp16;
                    out_ptr0[static_cast<long>(x1 + (1728L*x0))] = tmp15;
                    out_ptr1[static_cast<long>(x1 + (192L*x0))] = tmp17;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(2L*x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(14);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>(2L*x2);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>(x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : -std::numeric_limits<decltype(tmp11())>::infinity();
                            auto tmp14 = c10::convert<long>(1L + (2L*x2));
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = out_ptr0[static_cast<long>(768L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : -std::numeric_limits<decltype(tmp19())>::infinity();
                            auto tmp22 = max_propagate_nan(tmp21, tmp13);
                            auto tmp23 = c10::convert<long>(2L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = out_ptr0[static_cast<long>(1536L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : -std::numeric_limits<decltype(tmp28())>::infinity();
                            auto tmp31 = max_propagate_nan(tmp30, tmp22);
                            auto tmp32 = c10::convert<long>(1L + (2L*x1));
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = out_ptr0[static_cast<long>(10752L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : -std::numeric_limits<decltype(tmp37())>::infinity();
                            auto tmp40 = max_propagate_nan(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = out_ptr0[static_cast<long>(11520L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = max_propagate_nan(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = out_ptr0[static_cast<long>(12288L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : -std::numeric_limits<decltype(tmp47())>::infinity();
                            auto tmp50 = max_propagate_nan(tmp49, tmp45);
                            auto tmp51 = c10::convert<long>(2L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = out_ptr0[static_cast<long>(21504L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : -std::numeric_limits<decltype(tmp56())>::infinity();
                            auto tmp59 = max_propagate_nan(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = out_ptr0[static_cast<long>(22272L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : -std::numeric_limits<decltype(tmp61())>::infinity();
                            auto tmp64 = max_propagate_nan(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = out_ptr0[static_cast<long>(23040L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : -std::numeric_limits<decltype(tmp66())>::infinity();
                            auto tmp69 = max_propagate_nan(tmp68, tmp64);
                            auto tmp70 = [&]
                            {
                                auto tmp71 = out_ptr0[static_cast<long>(x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp10 ? tmp70() : -std::numeric_limits<decltype(tmp70())>::infinity();
                            auto tmp73 = [&]
                            {
                                auto tmp74 = out_ptr0[static_cast<long>(768L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp74;
                            }
                            ;
                            auto tmp75 = tmp18 ? tmp73() : -std::numeric_limits<decltype(tmp73())>::infinity();
                            auto tmp76 = tmp75 > tmp72;
                            auto tmp77 = c10::convert<long>(1L + (2L*x2) + (28L*x1));
                            auto tmp78 = c10::convert<long>((2L*x2) + (28L*x1));
                            auto tmp79 = tmp76 ? tmp77 : tmp78;
                            auto tmp80 = max_propagate_nan(tmp75, tmp72);
                            auto tmp81 = [&]
                            {
                                auto tmp82 = out_ptr0[static_cast<long>(1536L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp82;
                            }
                            ;
                            auto tmp83 = tmp27 ? tmp81() : -std::numeric_limits<decltype(tmp81())>::infinity();
                            auto tmp84 = tmp83 > tmp80;
                            auto tmp85 = c10::convert<long>(2L + (2L*x2) + (28L*x1));
                            auto tmp86 = tmp84 ? tmp85 : tmp79;
                            auto tmp87 = max_propagate_nan(tmp83, tmp80);
                            auto tmp88 = [&]
                            {
                                auto tmp89 = out_ptr0[static_cast<long>(10752L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp89;
                            }
                            ;
                            auto tmp90 = tmp36 ? tmp88() : -std::numeric_limits<decltype(tmp88())>::infinity();
                            auto tmp91 = tmp90 > tmp87;
                            auto tmp92 = c10::convert<long>(14L + (2L*x2) + (28L*x1));
                            auto tmp93 = tmp91 ? tmp92 : tmp86;
                            auto tmp94 = max_propagate_nan(tmp90, tmp87);
                            auto tmp95 = [&]
                            {
                                auto tmp96 = out_ptr0[static_cast<long>(11520L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp96;
                            }
                            ;
                            auto tmp97 = tmp41 ? tmp95() : -std::numeric_limits<decltype(tmp95())>::infinity();
                            auto tmp98 = tmp97 > tmp94;
                            auto tmp99 = c10::convert<long>(15L + (2L*x2) + (28L*x1));
                            auto tmp100 = tmp98 ? tmp99 : tmp93;
                            auto tmp101 = max_propagate_nan(tmp97, tmp94);
                            auto tmp102 = [&]
                            {
                                auto tmp103 = out_ptr0[static_cast<long>(12288L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp46 ? tmp102() : -std::numeric_limits<decltype(tmp102())>::infinity();
                            auto tmp105 = tmp104 > tmp101;
                            auto tmp106 = c10::convert<long>(16L + (2L*x2) + (28L*x1));
                            auto tmp107 = tmp105 ? tmp106 : tmp100;
                            auto tmp108 = max_propagate_nan(tmp104, tmp101);
                            auto tmp109 = [&]
                            {
                                auto tmp110 = out_ptr0[static_cast<long>(21504L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp110;
                            }
                            ;
                            auto tmp111 = tmp55 ? tmp109() : -std::numeric_limits<decltype(tmp109())>::infinity();
                            auto tmp112 = tmp111 > tmp108;
                            auto tmp113 = c10::convert<long>(28L + (2L*x2) + (28L*x1));
                            auto tmp114 = tmp112 ? tmp113 : tmp107;
                            auto tmp115 = max_propagate_nan(tmp111, tmp108);
                            auto tmp116 = [&]
                            {
                                auto tmp117 = out_ptr0[static_cast<long>(22272L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp117;
                            }
                            ;
                            auto tmp118 = tmp60 ? tmp116() : -std::numeric_limits<decltype(tmp116())>::infinity();
                            auto tmp119 = tmp118 > tmp115;
                            auto tmp120 = c10::convert<long>(29L + (2L*x2) + (28L*x1));
                            auto tmp121 = tmp119 ? tmp120 : tmp114;
                            auto tmp122 = max_propagate_nan(tmp118, tmp115);
                            auto tmp123 = [&]
                            {
                                auto tmp124 = out_ptr0[static_cast<long>(23040L + x3 + (1536L*x2) + (21504L*x1) + (150528L*x0))];
                                return tmp124;
                            }
                            ;
                            auto tmp125 = tmp65 ? tmp123() : -std::numeric_limits<decltype(tmp123())>::infinity();
                            auto tmp126 = tmp125 > tmp122;
                            auto tmp127 = c10::convert<long>(30L + (2L*x2) + (28L*x1));
                            auto tmp128 = tmp126 ? tmp127 : tmp121;
                            auto tmp129 = max_propagate_nan(tmp125, tmp122);
                            out_ptr1[static_cast<long>(x3 + (768L*x2) + (5376L*x1) + (37632L*x0))] = tmp69;
                            out_ptr2[static_cast<long>(x3 + (768L*x2) + (5376L*x1) + (37632L*x0))] = tmp128;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                tmp17.store(out_ptr1 + static_cast<long>(x1 + (1888L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                tmp17.store(out_ptr1 + static_cast<long>(x1 + (1888L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                tmp17.store(out_ptr1 + static_cast<long>(x1 + (1888L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                tmp17.store(out_ptr1 + static_cast<long>(x1 + (1888L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_threshold_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       bool* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (224L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                auto tmp16 = static_cast<float>(0.0);
                auto tmp17 = tmp15 <= tmp16;
                out_ptr0[static_cast<long>(x1 + (1888L*x0))] = tmp15;
                out_ptr1[static_cast<long>(x1 + (224L*x0))] = tmp17;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (1888L*x2) + (92512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (2144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                tmp17.store(out_ptr1 + static_cast<long>(x1 + (2144L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                tmp17.store(out_ptr1 + static_cast<long>(x1 + (2144L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                tmp17.store(out_ptr1 + static_cast<long>(x1 + (2144L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.sqrt();
                auto tmp8 = tmp7.reciprocal();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp2 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                tmp17.store(out_ptr1 + static_cast<long>(x1 + (2144L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (224L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1)];
                auto tmp3 = in_ptr2[static_cast<long>(x1)];
                auto tmp11 = in_ptr3[static_cast<long>(x1)];
                auto tmp13 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = std::sqrt(tmp5);
                auto tmp7 = 1 / tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = decltype(tmp2)(tmp2 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = tmp14 * (tmp14>0);
                auto tmp16 = static_cast<float>(0.0);
                auto tmp17 = tmp15 <= tmp16;
                out_ptr0[static_cast<long>(x1 + (2144L*x0))] = tmp15;
                out_ptr1[static_cast<long>(x1 + (224L*x0))] = tmp17;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_view_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_threshold_backward_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198 = args
    args.clear()
    assert_size_stride(primals_1, (64, ), (1, ))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_20, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_22, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_24, (160, ), (1, ))
    assert_size_stride(primals_25, (160, ), (1, ))
    assert_size_stride(primals_26, (160, ), (1, ))
    assert_size_stride(primals_27, (160, ), (1, ))
    assert_size_stride(primals_28, (160, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_34, (192, ), (1, ))
    assert_size_stride(primals_35, (192, ), (1, ))
    assert_size_stride(primals_36, (192, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (192, ), (1, ))
    assert_size_stride(primals_44, (192, ), (1, ))
    assert_size_stride(primals_45, (192, ), (1, ))
    assert_size_stride(primals_46, (192, ), (1, ))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (192, ), (1, ))
    assert_size_stride(primals_50, (192, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_52, (192, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (224, ), (1, ))
    assert_size_stride(primals_56, (224, ), (1, ))
    assert_size_stride(primals_57, (224, ), (1, ))
    assert_size_stride(primals_58, (224, ), (1, ))
    assert_size_stride(primals_59, (224, ), (1, ))
    assert_size_stride(primals_60, (224, ), (1, ))
    assert_size_stride(primals_61, (224, ), (1, ))
    assert_size_stride(primals_62, (224, ), (1, ))
    assert_size_stride(primals_63, (224, ), (1, ))
    assert_size_stride(primals_64, (224, ), (1, ))
    assert_size_stride(primals_65, (1024, ), (1, ))
    assert_size_stride(primals_66, (1024, ), (1, ))
    assert_size_stride(primals_67, (224, ), (1, ))
    assert_size_stride(primals_68, (224, ), (1, ))
    assert_size_stride(primals_69, (224, ), (1, ))
    assert_size_stride(primals_70, (224, ), (1, ))
    assert_size_stride(primals_71, (224, ), (1, ))
    assert_size_stride(primals_72, (224, ), (1, ))
    assert_size_stride(primals_73, (224, ), (1, ))
    assert_size_stride(primals_74, (224, ), (1, ))
    assert_size_stride(primals_75, (224, ), (1, ))
    assert_size_stride(primals_76, (224, ), (1, ))
    assert_size_stride(primals_77, (1024, ), (1, ))
    assert_size_stride(primals_78, (1024, ), (1, ))
    assert_size_stride(primals_79, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_80, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_81, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_82, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_83, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_84, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_85, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_86, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_87, (256, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_88, (160, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_89, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_90, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_91, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_92, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_93, (512, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_94, (192, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_95, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_96, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_97, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_98, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_99, (768, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(primals_100, (192, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_101, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_102, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_103, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_104, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_105, (768, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(primals_106, (224, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_107, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_108, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_109, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_110, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_111, (1024, 1888, 1, 1), (1888, 1, 1, 1))
    assert_size_stride(primals_112, (224, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_113, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_114, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_115, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_116, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_117, (1024, 2144, 1, 1), (2144, 1, 1, 1))
    assert_size_stride(primals_118, (1000, 1024), (1024, 1))
    assert_size_stride(primals_119, (1000, ), (1, ))
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (64, ), (1, ))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (160, ), (1, ))
    assert_size_stride(primals_139, (160, ), (1, ))
    assert_size_stride(primals_140, (160, ), (1, ))
    assert_size_stride(primals_141, (160, ), (1, ))
    assert_size_stride(primals_142, (160, ), (1, ))
    assert_size_stride(primals_143, (160, ), (1, ))
    assert_size_stride(primals_144, (160, ), (1, ))
    assert_size_stride(primals_145, (160, ), (1, ))
    assert_size_stride(primals_146, (160, ), (1, ))
    assert_size_stride(primals_147, (160, ), (1, ))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (192, ), (1, ))
    assert_size_stride(primals_151, (192, ), (1, ))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_153, (192, ), (1, ))
    assert_size_stride(primals_154, (192, ), (1, ))
    assert_size_stride(primals_155, (192, ), (1, ))
    assert_size_stride(primals_156, (192, ), (1, ))
    assert_size_stride(primals_157, (192, ), (1, ))
    assert_size_stride(primals_158, (192, ), (1, ))
    assert_size_stride(primals_159, (192, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, ), (1, ))
    assert_size_stride(primals_162, (192, ), (1, ))
    assert_size_stride(primals_163, (192, ), (1, ))
    assert_size_stride(primals_164, (192, ), (1, ))
    assert_size_stride(primals_165, (192, ), (1, ))
    assert_size_stride(primals_166, (192, ), (1, ))
    assert_size_stride(primals_167, (192, ), (1, ))
    assert_size_stride(primals_168, (192, ), (1, ))
    assert_size_stride(primals_169, (192, ), (1, ))
    assert_size_stride(primals_170, (192, ), (1, ))
    assert_size_stride(primals_171, (192, ), (1, ))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (224, ), (1, ))
    assert_size_stride(primals_175, (224, ), (1, ))
    assert_size_stride(primals_176, (224, ), (1, ))
    assert_size_stride(primals_177, (224, ), (1, ))
    assert_size_stride(primals_178, (224, ), (1, ))
    assert_size_stride(primals_179, (224, ), (1, ))
    assert_size_stride(primals_180, (224, ), (1, ))
    assert_size_stride(primals_181, (224, ), (1, ))
    assert_size_stride(primals_182, (224, ), (1, ))
    assert_size_stride(primals_183, (224, ), (1, ))
    assert_size_stride(primals_184, (1024, ), (1, ))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (224, ), (1, ))
    assert_size_stride(primals_187, (224, ), (1, ))
    assert_size_stride(primals_188, (224, ), (1, ))
    assert_size_stride(primals_189, (224, ), (1, ))
    assert_size_stride(primals_190, (224, ), (1, ))
    assert_size_stride(primals_191, (224, ), (1, ))
    assert_size_stride(primals_192, (224, ), (1, ))
    assert_size_stride(primals_193, (224, ), (1, ))
    assert_size_stride(primals_194, (224, ), (1, ))
    assert_size_stride(primals_195, (224, ), (1, ))
    assert_size_stride(primals_196, (1024, ), (1, ))
    assert_size_stride(primals_197, (1024, ), (1, ))
    assert_size_stride(primals_198, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((160, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((192, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((192, 768, 3, 3), (6912, 1, 2304, 768), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((224, 768, 3, 3), (6912, 1, 2304, 768), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    buf26 = empty_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    buf27 = empty_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((224, 1024, 3, 3), (9216, 1, 3072, 1024), device='cpu', dtype=torch.float32)
    buf29 = empty_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    buf30 = empty_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    buf32 = empty_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    buf33 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    del primals_100
    del primals_101
    del primals_102
    del primals_103
    del primals_104
    del primals_106
    del primals_107
    del primals_108
    del primals_109
    del primals_110
    del primals_112
    del primals_113
    del primals_114
    del primals_115
    del primals_116
    del primals_198
    del primals_79
    del primals_80
    del primals_81
    del primals_82
    del primals_83
    del primals_84
    del primals_85
    del primals_86
    del primals_88
    del primals_89
    del primals_90
    del primals_91
    del primals_92
    del primals_94
    del primals_95
    del primals_96
    del primals_97
    del primals_98
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf33, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf34, (4, 64, 112, 112), (802816, 1, 7168, 64))
    buf35 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf34.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf35.data_ptr()))
    del primals_2
    # Source Nodes: [x_5], Original ATen: [aten.convolution]
    buf36 = extern_kernels.convolution(buf35, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf36, (4, 64, 112, 112), (802816, 1, 7168, 64))
    buf37 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_2(c_void_p(buf36.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf37.data_ptr()))
    del primals_4
    # Source Nodes: [x_10], Original ATen: [aten.convolution]
    buf38 = extern_kernels.convolution(buf37, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf38, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf39 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf55 = empty_strided((4, 768, 56, 56), (2408448, 1, 43008, 768), device='cpu', dtype=torch.float32)
    buf50 = reinterpret_tensor(buf55, (4, 128, 56, 56), (2408448, 1, 43008, 768), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_3(c_void_p(buf38.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf50.data_ptr()))
    del primals_6
    # Source Nodes: [x_16], Original ATen: [aten.convolution]
    buf40 = extern_kernels.convolution(buf39, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf40, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf41 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf51 = reinterpret_tensor(buf55, (4, 128, 56, 56), (2408448, 1, 43008, 768), 128)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_4(c_void_p(buf40.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf51.data_ptr()))
    del primals_8
    # Source Nodes: [x_21], Original ATen: [aten.convolution]
    buf42 = extern_kernels.convolution(buf41, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf42, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf43 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf52 = reinterpret_tensor(buf55, (4, 128, 56, 56), (2408448, 1, 43008, 768), 256)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_5(c_void_p(buf42.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf52.data_ptr()))
    del primals_10
    # Source Nodes: [x_26], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf43, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf45 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf53 = reinterpret_tensor(buf55, (4, 128, 56, 56), (2408448, 1, 43008, 768), 384)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_6(c_void_p(buf44.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf53.data_ptr()))
    del primals_12
    # Source Nodes: [x_31], Original ATen: [aten.convolution]
    buf46 = extern_kernels.convolution(buf45, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf46, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf47 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf54 = reinterpret_tensor(buf55, (4, 128, 56, 56), (2408448, 1, 43008, 768), 512)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_7(c_void_p(buf46.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf54.data_ptr()))
    del primals_14
    # Source Nodes: [x_36], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(buf47, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf48, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf49 = reinterpret_tensor(buf55, (4, 128, 56, 56), (2408448, 1, 43008, 768), 640)  # alias
    buf163 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_8(c_void_p(buf48.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf163.data_ptr()))
    del primals_16
    # Source Nodes: [x_43], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (4, 256, 56, 56), (802816, 1, 14336, 256))
    buf57 = empty_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    buf58 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    buf59 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_9(c_void_p(buf56.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()))
    del primals_18
    # Source Nodes: [x_51], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(buf58, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (4, 160, 28, 28), (125440, 1, 4480, 160))
    buf61 = empty_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    buf75 = empty_strided((4, 1056, 28, 28), (827904, 1, 29568, 1056), device='cpu', dtype=torch.float32)
    buf71 = reinterpret_tensor(buf75, (4, 160, 28, 28), (827904, 1, 29568, 1056), 256)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_10(c_void_p(buf60.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf71.data_ptr()))
    del primals_20
    # Source Nodes: [x_56], Original ATen: [aten.convolution]
    buf62 = extern_kernels.convolution(buf61, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (4, 160, 28, 28), (125440, 1, 4480, 160))
    buf63 = empty_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    buf72 = reinterpret_tensor(buf75, (4, 160, 28, 28), (827904, 1, 29568, 1056), 416)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_11(c_void_p(buf62.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf72.data_ptr()))
    del primals_22
    # Source Nodes: [x_61], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf63, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf64, (4, 160, 28, 28), (125440, 1, 4480, 160))
    buf65 = empty_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    buf73 = reinterpret_tensor(buf75, (4, 160, 28, 28), (827904, 1, 29568, 1056), 576)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_12(c_void_p(buf64.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_24
    # Source Nodes: [x_66], Original ATen: [aten.convolution]
    buf66 = extern_kernels.convolution(buf65, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf66, (4, 160, 28, 28), (125440, 1, 4480, 160))
    buf67 = empty_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    buf74 = reinterpret_tensor(buf75, (4, 160, 28, 28), (827904, 1, 29568, 1056), 736)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_13(c_void_p(buf66.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf74.data_ptr()))
    del primals_26
    # Source Nodes: [x_71], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (4, 160, 28, 28), (125440, 1, 4480, 160))
    buf69 = reinterpret_tensor(buf75, (4, 160, 28, 28), (827904, 1, 29568, 1056), 896)  # alias
    buf162 = empty_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.bool)
    buf70 = reinterpret_tensor(buf75, (4, 256, 28, 28), (827904, 1, 29568, 1056), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_threshold_backward_14(c_void_p(buf68.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf70.data_ptr()))
    del primals_28
    # Source Nodes: [x_78], Original ATen: [aten.convolution]
    buf76 = extern_kernels.convolution(buf75, primals_93, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf76, (4, 512, 28, 28), (401408, 1, 14336, 512))
    buf77 = empty_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    buf78 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    buf79 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_15(c_void_p(buf76.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del primals_30
    # Source Nodes: [x_86], Original ATen: [aten.convolution]
    buf80 = extern_kernels.convolution(buf78, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf80, (4, 192, 14, 14), (37632, 1, 2688, 192))
    buf81 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    buf95 = empty_strided((4, 1472, 14, 14), (288512, 1, 20608, 1472), device='cpu', dtype=torch.float32)
    buf91 = reinterpret_tensor(buf95, (4, 192, 14, 14), (288512, 1, 20608, 1472), 512)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_16(c_void_p(buf80.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf91.data_ptr()))
    del primals_32
    # Source Nodes: [x_91], Original ATen: [aten.convolution]
    buf82 = extern_kernels.convolution(buf81, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf82, (4, 192, 14, 14), (37632, 1, 2688, 192))
    buf83 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    buf92 = reinterpret_tensor(buf95, (4, 192, 14, 14), (288512, 1, 20608, 1472), 704)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_17(c_void_p(buf82.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf92.data_ptr()))
    del primals_34
    # Source Nodes: [x_96], Original ATen: [aten.convolution]
    buf84 = extern_kernels.convolution(buf83, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (4, 192, 14, 14), (37632, 1, 2688, 192))
    buf85 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    buf93 = reinterpret_tensor(buf95, (4, 192, 14, 14), (288512, 1, 20608, 1472), 896)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_18(c_void_p(buf84.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf93.data_ptr()))
    del primals_36
    # Source Nodes: [x_101], Original ATen: [aten.convolution]
    buf86 = extern_kernels.convolution(buf85, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (4, 192, 14, 14), (37632, 1, 2688, 192))
    buf87 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    buf94 = reinterpret_tensor(buf95, (4, 192, 14, 14), (288512, 1, 20608, 1472), 1088)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_19(c_void_p(buf86.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf94.data_ptr()))
    del primals_38
    # Source Nodes: [x_106], Original ATen: [aten.convolution]
    buf88 = extern_kernels.convolution(buf87, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf88, (4, 192, 14, 14), (37632, 1, 2688, 192))
    buf89 = reinterpret_tensor(buf95, (4, 192, 14, 14), (288512, 1, 20608, 1472), 1280)  # alias
    buf161 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.bool)
    buf90 = reinterpret_tensor(buf95, (4, 512, 14, 14), (288512, 1, 20608, 1472), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_threshold_backward_20(c_void_p(buf88.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf90.data_ptr()))
    del primals_40
    # Source Nodes: [x_113], Original ATen: [aten.convolution]
    buf96 = extern_kernels.convolution(buf95, primals_99, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf96, (4, 768, 14, 14), (150528, 1, 10752, 768))
    buf97 = empty_strided((4, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    buf113 = empty_strided((4, 1728, 14, 14), (338688, 1, 24192, 1728), device='cpu', dtype=torch.float32)
    buf108 = reinterpret_tensor(buf113, (4, 768, 14, 14), (338688, 1, 24192, 1728), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_21(c_void_p(buf96.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf108.data_ptr()))
    del primals_42
    # Source Nodes: [x_119], Original ATen: [aten.convolution]
    buf98 = extern_kernels.convolution(buf97, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf98, (4, 192, 14, 14), (37632, 1, 2688, 192))
    buf99 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    buf109 = reinterpret_tensor(buf113, (4, 192, 14, 14), (338688, 1, 24192, 1728), 768)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_22(c_void_p(buf98.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf109.data_ptr()))
    del primals_44
    # Source Nodes: [x_124], Original ATen: [aten.convolution]
    buf100 = extern_kernels.convolution(buf99, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf100, (4, 192, 14, 14), (37632, 1, 2688, 192))
    buf101 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    buf110 = reinterpret_tensor(buf113, (4, 192, 14, 14), (338688, 1, 24192, 1728), 960)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_23(c_void_p(buf100.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf110.data_ptr()))
    del primals_46
    # Source Nodes: [x_129], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (4, 192, 14, 14), (37632, 1, 2688, 192))
    buf103 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    buf111 = reinterpret_tensor(buf113, (4, 192, 14, 14), (338688, 1, 24192, 1728), 1152)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_24(c_void_p(buf102.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf111.data_ptr()))
    del primals_48
    # Source Nodes: [x_134], Original ATen: [aten.convolution]
    buf104 = extern_kernels.convolution(buf103, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf104, (4, 192, 14, 14), (37632, 1, 2688, 192))
    buf105 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    buf112 = reinterpret_tensor(buf113, (4, 192, 14, 14), (338688, 1, 24192, 1728), 1344)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_25(c_void_p(buf104.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf112.data_ptr()))
    del primals_50
    # Source Nodes: [x_139], Original ATen: [aten.convolution]
    buf106 = extern_kernels.convolution(buf105, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf106, (4, 192, 14, 14), (37632, 1, 2688, 192))
    buf107 = reinterpret_tensor(buf113, (4, 192, 14, 14), (338688, 1, 24192, 1728), 1536)  # alias
    buf160 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_26(c_void_p(buf106.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf160.data_ptr()))
    del primals_52
    # Source Nodes: [x_146], Original ATen: [aten.convolution]
    buf114 = extern_kernels.convolution(buf113, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (4, 768, 14, 14), (150528, 1, 10752, 768))
    buf115 = empty_strided((4, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    buf116 = empty_strided((4, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    buf117 = empty_strided((4, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_27(c_void_p(buf114.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del primals_54
    # Source Nodes: [x_154], Original ATen: [aten.convolution]
    buf118 = extern_kernels.convolution(buf116, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf118, (4, 224, 7, 7), (10976, 1, 1568, 224))
    buf119 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    buf133 = empty_strided((4, 1888, 7, 7), (92512, 1, 13216, 1888), device='cpu', dtype=torch.float32)
    buf129 = reinterpret_tensor(buf133, (4, 224, 7, 7), (92512, 1, 13216, 1888), 768)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_28(c_void_p(buf118.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf129.data_ptr()))
    del primals_56
    # Source Nodes: [x_159], Original ATen: [aten.convolution]
    buf120 = extern_kernels.convolution(buf119, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf120, (4, 224, 7, 7), (10976, 1, 1568, 224))
    buf121 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    buf130 = reinterpret_tensor(buf133, (4, 224, 7, 7), (92512, 1, 13216, 1888), 992)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_29(c_void_p(buf120.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf130.data_ptr()))
    del primals_58
    # Source Nodes: [x_164], Original ATen: [aten.convolution]
    buf122 = extern_kernels.convolution(buf121, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf122, (4, 224, 7, 7), (10976, 1, 1568, 224))
    buf123 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    buf131 = reinterpret_tensor(buf133, (4, 224, 7, 7), (92512, 1, 13216, 1888), 1216)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_30(c_void_p(buf122.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf131.data_ptr()))
    del primals_60
    # Source Nodes: [x_169], Original ATen: [aten.convolution]
    buf124 = extern_kernels.convolution(buf123, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (4, 224, 7, 7), (10976, 1, 1568, 224))
    buf125 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    buf132 = reinterpret_tensor(buf133, (4, 224, 7, 7), (92512, 1, 13216, 1888), 1440)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_31(c_void_p(buf124.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()))
    del primals_62
    # Source Nodes: [x_174], Original ATen: [aten.convolution]
    buf126 = extern_kernels.convolution(buf125, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf126, (4, 224, 7, 7), (10976, 1, 1568, 224))
    buf127 = reinterpret_tensor(buf133, (4, 224, 7, 7), (92512, 1, 13216, 1888), 1664)  # alias
    buf159 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.bool)
    buf128 = reinterpret_tensor(buf133, (4, 768, 7, 7), (92512, 1, 13216, 1888), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_threshold_backward_32(c_void_p(buf126.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf128.data_ptr()))
    del primals_64
    # Source Nodes: [x_181], Original ATen: [aten.convolution]
    buf134 = extern_kernels.convolution(buf133, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf134, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    buf135 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    buf151 = empty_strided((4, 2144, 7, 7), (105056, 1, 15008, 2144), device='cpu', dtype=torch.float32)
    buf146 = reinterpret_tensor(buf151, (4, 1024, 7, 7), (105056, 1, 15008, 2144), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_33(c_void_p(buf134.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf146.data_ptr()))
    del primals_66
    # Source Nodes: [x_187], Original ATen: [aten.convolution]
    buf136 = extern_kernels.convolution(buf135, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf136, (4, 224, 7, 7), (10976, 1, 1568, 224))
    buf137 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    buf147 = reinterpret_tensor(buf151, (4, 224, 7, 7), (105056, 1, 15008, 2144), 1024)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_34(c_void_p(buf136.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf147.data_ptr()))
    del primals_68
    # Source Nodes: [x_192], Original ATen: [aten.convolution]
    buf138 = extern_kernels.convolution(buf137, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf138, (4, 224, 7, 7), (10976, 1, 1568, 224))
    buf139 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    buf148 = reinterpret_tensor(buf151, (4, 224, 7, 7), (105056, 1, 15008, 2144), 1248)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_35(c_void_p(buf138.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf148.data_ptr()))
    del primals_70
    # Source Nodes: [x_197], Original ATen: [aten.convolution]
    buf140 = extern_kernels.convolution(buf139, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf140, (4, 224, 7, 7), (10976, 1, 1568, 224))
    buf141 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    buf149 = reinterpret_tensor(buf151, (4, 224, 7, 7), (105056, 1, 15008, 2144), 1472)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_36(c_void_p(buf140.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf149.data_ptr()))
    del primals_72
    # Source Nodes: [x_202], Original ATen: [aten.convolution]
    buf142 = extern_kernels.convolution(buf141, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf142, (4, 224, 7, 7), (10976, 1, 1568, 224))
    buf143 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    buf150 = reinterpret_tensor(buf151, (4, 224, 7, 7), (105056, 1, 15008, 2144), 1696)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_37(c_void_p(buf142.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf150.data_ptr()))
    del primals_74
    # Source Nodes: [x_207], Original ATen: [aten.convolution]
    buf144 = extern_kernels.convolution(buf143, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf144, (4, 224, 7, 7), (10976, 1, 1568, 224))
    buf145 = reinterpret_tensor(buf151, (4, 224, 7, 7), (105056, 1, 15008, 2144), 1920)  # alias
    buf158 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_no_training_relu_threshold_backward_38(c_void_p(buf144.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf158.data_ptr()))
    del primals_76
    # Source Nodes: [x_214], Original ATen: [aten.convolution]
    buf152 = extern_kernels.convolution(buf151, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf152, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    buf153 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    buf154 = empty_strided((4, 1024, 1, 1), (1024, 1, 4096, 4096), device='cpu', dtype=torch.float32)
    buf155 = reinterpret_tensor(buf154, (4, 1024), (1024, 1), 0); del buf154  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_view_39(c_void_p(buf155.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf153.data_ptr()))
    del primals_78
    buf156 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_226], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_119, buf155, reinterpret_tensor(primals_118, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf156)
    del primals_119
    buf157 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.bool)
    cpp_fused_threshold_backward_40(c_void_p(buf153.data_ptr()), c_void_p(buf157.data_ptr()))
    return (buf156, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, primals_87, buf8, buf9, buf10, buf11, buf12, primals_93, buf13, buf14, buf15, buf16, buf17, primals_99, buf18, buf19, buf20, buf21, buf22, primals_105, buf23, buf24, buf25, buf26, buf27, primals_111, buf28, buf29, buf30, buf31, buf32, primals_117, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf113, buf114, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf133, buf134, buf135, buf136, buf137, buf138, buf139, buf140, buf141, buf142, buf143, buf144, buf151, buf152, buf155, reinterpret_tensor(primals_118, (1000, 1024), (1024, 1), 0), buf157, buf158, buf159, buf160, buf161, buf162, buf163, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((256, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((160, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, 1056, 1, 1), (1056, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((192, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, 1472, 1, 1), (1472, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((192, 768, 3, 3), (6912, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, 1728, 1, 1), (1728, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((224, 768, 3, 3), (6912, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1024, 1888, 1, 1), (1888, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((224, 1024, 3, 3), (9216, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1024, 2144, 1, 1), (2144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_vovnet', benchmark_compiled_module)
