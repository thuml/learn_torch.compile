
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


cpp_fused__native_batch_norm_legit_view_0 = async_compile.cpp('''
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
                       float* out_ptr32)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr6 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr9 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr9[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr9 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr10 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr10[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr11 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr11[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr12 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr12[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr13 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr13[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr14 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr14[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr15 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr15[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr16 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr16[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr17 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr17[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr18 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr18[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr19 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr19[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr20 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr20[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr21 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr21[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr22 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr22[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr23 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr23[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr24 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr24[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr25 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr25[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr26 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr26[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr27 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr27[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr28[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr28[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(27L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((3L*(static_cast<long>(x1) % static_cast<long>(9L))) + (27L*x0) + (27L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr29 + static_cast<long>(x0));
                        tmp_acc0_vec.m2.store(out_ptr30 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(27.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr31 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(x2 + (3L*x1) + (27L*x0))];
                            auto tmp1 = out_ptr29[static_cast<long>(x0)];
                            auto tmp3 = out_ptr30[static_cast<long>(x0)];
                            auto tmp10 = in_ptr29[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(27.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            auto tmp11 = static_cast<float>(0.34412564994580647);
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = decltype(tmp9)(tmp9 * tmp12);
                            out_ptr32[static_cast<long>(x2 + (3L*x1) + (27L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(9L))) + (144L*x0) + (144L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (16L*x1) + (144L*x0)));
                            auto tmp1 = out_ptr1[static_cast<long>(x0)];
                            auto tmp4 = out_ptr2[static_cast<long>(x0)];
                            auto tmp12 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(144.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.1490107774734497);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp11 * tmp15;
                            tmp16.store(out_ptr4 + static_cast<long>(x2 + (16L*x1) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((32L*(static_cast<long>(x1) % static_cast<long>(9L))) + (288L*x0) + (288L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(288.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x1) + (288L*x0)));
                            auto tmp1 = out_ptr1[static_cast<long>(x0)];
                            auto tmp4 = out_ptr2[static_cast<long>(x0)];
                            auto tmp12 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(288.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.10536653122135592);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp11 * tmp15;
                            tmp16.store(out_ptr4 + static_cast<long>(x2 + (32L*x1) + (288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                            auto tmp1 = out_ptr1[static_cast<long>(x0)];
                            auto tmp4 = out_ptr2[static_cast<long>(x0)];
                            auto tmp12 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(576.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.07450538873672485);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp11 * tmp15;
                            tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_mul_silu_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                tmp5.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(128.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.1580497968320339);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = in_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = static_cast<float>(0.1580497968320339);
                auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp11 * tmp15;
                tmp16.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                            auto tmp1 = out_ptr1[static_cast<long>(x0)];
                            auto tmp4 = out_ptr2[static_cast<long>(x0)];
                            auto tmp12 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(576.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.07450538873672485);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp11 * tmp15;
                            tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                            auto tmp1 = out_ptr1[static_cast<long>(x0)];
                            auto tmp4 = out_ptr2[static_cast<long>(x0)];
                            auto tmp12 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(576.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.07450538873672485);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp11 * tmp15;
                            tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(64.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.22351616621017456);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mean_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(3136.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_avg_pool2d_mul_sigmoid_silu_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp14 = static_cast<float>(0.9805806756909201);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(14336L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(14592L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (7168L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = out_ptr3[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.11175808310508728);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr5 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(256.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = in_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(256.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = static_cast<float>(0.11175808310508728);
                auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp11 * tmp15;
                tmp16.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                            auto tmp1 = out_ptr1[static_cast<long>(x0)];
                            auto tmp4 = out_ptr2[static_cast<long>(x0)];
                            auto tmp12 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(576.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.07450538873672485);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp11 * tmp15;
                            tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                            auto tmp1 = out_ptr1[static_cast<long>(x0)];
                            auto tmp4 = out_ptr2[static_cast<long>(x0)];
                            auto tmp12 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(576.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.07450538873672485);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp11 * tmp15;
                            tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(128.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.1580497968320339);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mean_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (401408L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp14 = static_cast<float>(0.9805806756909201);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(512.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07902489841601695);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                            auto tmp1 = out_ptr1[static_cast<long>(x0)];
                            auto tmp4 = out_ptr2[static_cast<long>(x0)];
                            auto tmp12 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(576.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.07450538873672485);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp11 * tmp15;
                            tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                            auto tmp1 = out_ptr1[static_cast<long>(x0)];
                            auto tmp4 = out_ptr2[static_cast<long>(x0)];
                            auto tmp12 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(576.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.07450538873672485);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp11 * tmp15;
                            tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(128.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.1580497968320339);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mean_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (401408L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_avg_pool2d_mul_sigmoid_silu_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp14 = tmp13 * tmp5;
                        auto tmp15 = tmp14 * tmp8;
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp18 = tmp9 + tmp17;
                        tmp18.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(0.9622504486493761);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                tmp5.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(512L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(14336L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(14848L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (7168L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr3[static_cast<long>(x0)];
                    auto tmp4 = out_ptr4[static_cast<long>(x0)];
                    auto tmp12 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.07902489841601695);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr6 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.07902489841601695);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.09125009274634042);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp14 = static_cast<float>(0.9805806756909201);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04562504637317021);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.09125009274634042);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp14 = tmp13 * tmp5;
                        auto tmp15 = tmp14 * tmp8;
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp18 = tmp9 + tmp17;
                        tmp18.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(0.9622504486493761);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                tmp5.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = out_ptr3[static_cast<long>(x0)];
                    auto tmp12 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04562504637317021);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.09125009274634042);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp14 = static_cast<float>(0.9449111825230679);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04562504637317021);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.09125009274634042);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp14 = tmp13 * tmp5;
                        auto tmp15 = tmp14 * tmp8;
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp18 = tmp9 + tmp17;
                        tmp18.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(0.9284766908852592);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                tmp5.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = out_ptr3[static_cast<long>(x0)];
                    auto tmp12 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04562504637317021);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.09125009274634042);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp14 = static_cast<float>(0.9128709291752768);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04562504637317021);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.09125009274634042);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_avg_pool2d_mul_sigmoid_silu_view_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp14 = tmp13 * tmp5;
                        auto tmp15 = tmp14 * tmp8;
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp18 = tmp9 + tmp17;
                        tmp18.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(0.8980265101338745);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                tmp5.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (3072L*x1) + (43008L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(1536L + x2 + (3072L*x1) + (43008L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(21504L + x2 + (3072L*x1) + (43008L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(23040L + x2 + (3072L*x1) + (43008L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr2 + static_cast<long>(x2 + (1536L*x1) + (10752L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr3[static_cast<long>(x0)];
                    auto tmp4 = out_ptr4[static_cast<long>(x0)];
                    auto tmp12 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04562504637317021);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr6 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04562504637317021);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.09125009274634042);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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


cpp_fused_relu_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp14 = static_cast<float>(0.9805806756909201);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04562504637317021);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.09125009274634042);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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


cpp_fused_relu_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_fill_mul_sigmoid_silu_sub_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp14 = tmp13 * tmp5;
                        auto tmp15 = tmp14 * tmp8;
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp18 = tmp9 + tmp17;
                        auto tmp19 = decltype(tmp17)(1)/(decltype(tmp17)(1) + tmp17.neg().exp());
                        auto tmp20 = static_cast<float>(1.0);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp21 - tmp19;
                        auto tmp23 = tmp17 * tmp22;
                        auto tmp24 = tmp23 + tmp21;
                        auto tmp25 = tmp19 * tmp24;
                        tmp18.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        tmp25.store(out_ptr1 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(0.9622504486493761);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp7 - tmp1;
                auto tmp9 = tmp0 * tmp8;
                auto tmp10 = tmp9 + tmp7;
                auto tmp11 = tmp1 * tmp10;
                tmp5.store(out_ptr2 + static_cast<long>(x0));
                tmp11.store(out_ptr3 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr4[static_cast<long>(x0)];
                    auto tmp4 = out_ptr5[static_cast<long>(x0)];
                    auto tmp12 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04562504637317021);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr7 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(576.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.07450538873672485);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_silu_view_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.09125009274634042);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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


cpp_fused_relu_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_mul_sigmoid_view_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1536.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04562504637317021);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_silu_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2304L*x2) + (112896L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (2304L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_fill_mul_sigmoid_silu_sub_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       float* in_out_ptr4,
                       float* in_out_ptr5,
                       float* in_out_ptr6,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr2 = in_out_ptr1;
    auto in_ptr5 = in_out_ptr2;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = static_cast<float>(1.0);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp14 - tmp12;
                        auto tmp16 = tmp11 * tmp15;
                        auto tmp17 = tmp16 + tmp14;
                        auto tmp18 = tmp12 * tmp17;
                        tmp18.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = static_cast<float>(1.0);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp14 - tmp12;
                        auto tmp16 = tmp11 * tmp15;
                        auto tmp17 = tmp16 + tmp14;
                        auto tmp18 = tmp12 * tmp17;
                        tmp18.store(out_ptr1 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = static_cast<float>(1.0);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp14 - tmp12;
                        auto tmp16 = tmp11 * tmp15;
                        auto tmp17 = tmp16 + tmp14;
                        auto tmp18 = tmp12 * tmp17;
                        tmp18.store(in_out_ptr3 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = static_cast<float>(1.0);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp14 - tmp12;
                        auto tmp16 = tmp11 * tmp15;
                        auto tmp17 = tmp16 + tmp14;
                        auto tmp18 = tmp12 * tmp17;
                        tmp18.store(in_out_ptr5 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(0.2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp11)(1)/(decltype(tmp11)(1) + tmp11.neg().exp());
                        auto tmp13 = static_cast<float>(1.0);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp14 - tmp12;
                        auto tmp16 = tmp11 * tmp15;
                        auto tmp17 = tmp16 + tmp14;
                        auto tmp18 = tmp12 * tmp17;
                        tmp18.store(in_out_ptr6 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_5, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_14, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_20, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_32, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_35, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_38, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_44, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_47, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_50, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_57, (1536, ), (1, ))
    assert_size_stride(primals_58, (384, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_59, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_60, (384, ), (1, ))
    assert_size_stride(primals_61, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_62, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_65, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_68, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_69, (1536, ), (1, ))
    assert_size_stride(primals_70, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_71, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_72, (384, ), (1, ))
    assert_size_stride(primals_73, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_74, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_77, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_80, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_81, (1536, ), (1, ))
    assert_size_stride(primals_82, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_83, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_86, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_89, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_92, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_93, (1536, ), (1, ))
    assert_size_stride(primals_94, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_95, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_98, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_100, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_101, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_104, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_105, (1536, ), (1, ))
    assert_size_stride(primals_106, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_107, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_108, (384, ), (1, ))
    assert_size_stride(primals_109, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_110, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_113, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_115, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_116, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_117, (1536, ), (1, ))
    assert_size_stride(primals_118, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_119, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_120, (384, ), (1, ))
    assert_size_stride(primals_121, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_122, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_125, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_126, (384, ), (1, ))
    assert_size_stride(primals_127, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_128, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_129, (1536, ), (1, ))
    assert_size_stride(primals_130, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_131, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_132, (1536, ), (1, ))
    assert_size_stride(primals_133, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_134, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_137, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_140, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_143, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_144, (1536, ), (1, ))
    assert_size_stride(primals_145, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_146, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_149, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_152, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_155, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_156, (1536, ), (1, ))
    assert_size_stride(primals_157, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_158, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_161, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_164, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_167, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_168, (1536, ), (1, ))
    assert_size_stride(primals_169, (2304, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_170, (2304, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_171, (2304, ), (1, ))
    assert_size_stride(primals_172, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_183, (512, ), (1, ))
    assert_size_stride(primals_184, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_185, (384, ), (1, ))
    assert_size_stride(primals_186, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_187, (1536, ), (1, ))
    assert_size_stride(primals_188, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_190, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_191, (1536, ), (1, ))
    assert_size_stride(primals_192, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_194, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_195, (1536, ), (1, ))
    assert_size_stride(primals_196, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_197, (384, ), (1, ))
    assert_size_stride(primals_198, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_199, (1536, ), (1, ))
    assert_size_stride(primals_200, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_203, (1536, ), (1, ))
    assert_size_stride(primals_204, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_206, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_207, (1536, ), (1, ))
    assert_size_stride(primals_208, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_209, (384, ), (1, ))
    assert_size_stride(primals_210, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_211, (1536, ), (1, ))
    assert_size_stride(primals_212, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_213, (384, ), (1, ))
    assert_size_stride(primals_214, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_215, (1536, ), (1, ))
    assert_size_stride(primals_216, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_217, (384, ), (1, ))
    assert_size_stride(primals_218, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_219, (1536, ), (1, ))
    assert_size_stride(primals_220, (1000, 2304), (2304, 1))
    assert_size_stride(primals_221, (1000, ), (1, ))
    assert_size_stride(primals_222, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf26 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf27 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf29 = empty_strided((1, 16, 1), (16, 1, 16), device='cpu', dtype=torch.float32)
    buf30 = empty_strided((1, 16, 1), (16, 1, 16), device='cpu', dtype=torch.float32)
    buf32 = empty((16, ), device='cpu', dtype=torch.float32)
    buf33 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_view_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    del buf30
    del primals_1
    del primals_10
    del primals_100
    del primals_109
    del primals_112
    del primals_121
    del primals_124
    del primals_136
    del primals_139
    del primals_148
    del primals_151
    del primals_160
    del primals_163
    del primals_19
    del primals_22
    del primals_222
    del primals_34
    del primals_37
    del primals_4
    del primals_46
    del primals_49
    del primals_61
    del primals_64
    del primals_7
    del primals_73
    del primals_76
    del primals_85
    del primals_88
    del primals_97
    # Source Nodes: [conv2d], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf28, buf33, primals_3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf34, (8, 16, 112, 112), (200704, 1, 1792, 16))
    del primals_3
    buf35 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((1, 32, 1), (32, 1, 32), device='cpu', dtype=torch.float32)
    buf37 = empty_strided((1, 32, 1), (32, 1, 32), device='cpu', dtype=torch.float32)
    buf39 = empty((32, ), device='cpu', dtype=torch.float32)
    buf40 = empty_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_1(c_void_p(buf34.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    del buf37
    # Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
    buf41 = extern_kernels.convolution(buf35, buf40, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf41, (8, 32, 112, 112), (401408, 1, 3584, 32))
    del primals_6
    buf42 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    buf43 = empty_strided((1, 64, 1), (64, 1, 64), device='cpu', dtype=torch.float32)
    buf44 = empty_strided((1, 64, 1), (64, 1, 64), device='cpu', dtype=torch.float32)
    buf46 = empty((64, ), device='cpu', dtype=torch.float32)
    buf47 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_2(c_void_p(buf41.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    # Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(buf42, buf47, primals_9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf48, (8, 64, 112, 112), (802816, 1, 7168, 64))
    del primals_9
    buf49 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    buf50 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf51 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf53 = empty((128, ), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_3(c_void_p(buf48.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    # Source Nodes: [shortcut], Original ATen: [aten.convolution]
    buf55 = extern_kernels.convolution(buf49, buf54, primals_12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf55, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del primals_12
    buf56 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf57 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf58 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf60 = empty((256, ), device='cpu', dtype=torch.float32)
    buf61 = empty_strided((256, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_mul_silu_view_4(c_void_p(buf55.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    # Source Nodes: [shortcut_1], Original ATen: [aten.convolution]
    buf62 = extern_kernels.convolution(buf56, buf61, primals_15, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf62, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del primals_15
    buf63 = buf44; del buf44  # reuse
    buf64 = empty_strided((1, 64, 1), (64, 1, 64), device='cpu', dtype=torch.float32)
    buf66 = empty((64, ), device='cpu', dtype=torch.float32)
    buf67 = empty_strided((64, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_view_5(c_void_p(primals_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    # Source Nodes: [out_1], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf56, buf67, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf68, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del primals_18
    buf69 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    buf70 = buf64; del buf64  # reuse
    buf71 = empty_strided((1, 64, 1), (64, 1, 64), device='cpu', dtype=torch.float32)
    buf73 = empty((64, ), device='cpu', dtype=torch.float32)
    buf74 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_6(c_void_p(buf68.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    # Source Nodes: [out_2], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(buf69, buf74, primals_21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf75, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del primals_21
    buf76 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    buf77 = buf71; del buf71  # reuse
    buf78 = empty_strided((1, 64, 1), (64, 1, 64), device='cpu', dtype=torch.float32)
    buf80 = empty((64, ), device='cpu', dtype=torch.float32)
    buf81 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_7(c_void_p(buf75.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del buf78
    # Source Nodes: [out_3], Original ATen: [aten.convolution]
    buf82 = extern_kernels.convolution(buf76, buf81, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf82, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del primals_24
    buf83 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    buf84 = buf58; del buf58  # reuse
    buf85 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf87 = empty((256, ), device='cpu', dtype=torch.float32)
    buf88 = empty_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_8(c_void_p(buf82.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    del buf85
    # Source Nodes: [out_4], Original ATen: [aten.convolution]
    buf89 = extern_kernels.convolution(buf83, buf88, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf89, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del primals_27
    buf90 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf91 = reinterpret_tensor(buf90, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf90  # reuse
    cpp_fused_mean_9(c_void_p(buf91.data_ptr()), c_void_p(buf89.data_ptr()))
    # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, primals_172, primals_173, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf92, (8, 64, 1, 1), (64, 1, 64, 64))
    del primals_173
    buf93 = buf92; del buf92  # reuse
    cpp_fused_relu_10(c_void_p(buf93.data_ptr()))
    # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
    buf94 = extern_kernels.convolution(buf93, primals_174, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf94, (8, 256, 1, 1), (256, 1, 256, 256))
    del primals_175
    buf95 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    buf96 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf98 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf100 = empty((512, ), device='cpu', dtype=torch.float32)
    buf101 = empty_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_avg_pool2d_mul_sigmoid_silu_view_11(c_void_p(buf89.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    # Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf96, buf101, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf102, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del primals_30
    buf103 = buf51; del buf51  # reuse
    buf104 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf106 = empty((128, ), device='cpu', dtype=torch.float32)
    buf107 = empty_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_view_12(c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    # Source Nodes: [out_9], Original ATen: [aten.convolution]
    buf108 = extern_kernels.convolution(buf95, buf107, primals_33, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf108, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del primals_33
    buf109 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf110 = buf104; del buf104  # reuse
    buf111 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf113 = empty((128, ), device='cpu', dtype=torch.float32)
    buf114 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_13(c_void_p(buf108.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()))
    # Source Nodes: [out_10], Original ATen: [aten.convolution]
    buf115 = extern_kernels.convolution(buf109, buf114, primals_36, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2)
    assert_size_stride(buf115, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del primals_36
    buf116 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    buf117 = buf111; del buf111  # reuse
    buf118 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf120 = empty((128, ), device='cpu', dtype=torch.float32)
    buf121 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_14(c_void_p(buf115.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    # Source Nodes: [out_11], Original ATen: [aten.convolution]
    buf122 = extern_kernels.convolution(buf116, buf121, primals_39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2)
    assert_size_stride(buf122, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del primals_39
    buf123 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    buf124 = buf98; del buf98  # reuse
    buf125 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf127 = empty((512, ), device='cpu', dtype=torch.float32)
    buf128 = empty_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_15(c_void_p(buf122.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    # Source Nodes: [out_12], Original ATen: [aten.convolution]
    buf129 = extern_kernels.convolution(buf123, buf128, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf129, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del primals_42
    buf130 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cpu', dtype=torch.float32)
    buf131 = reinterpret_tensor(buf130, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf130  # reuse
    cpp_fused_mean_16(c_void_p(buf131.data_ptr()), c_void_p(buf129.data_ptr()))
    # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
    buf132 = extern_kernels.convolution(buf131, primals_176, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf132, (8, 128, 1, 1), (128, 1, 128, 128))
    del primals_177
    buf133 = buf132; del buf132  # reuse
    cpp_fused_relu_17(c_void_p(buf133.data_ptr()))
    # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
    buf134 = extern_kernels.convolution(buf133, primals_178, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf134, (8, 512, 1, 1), (512, 1, 512, 512))
    del primals_179
    buf135 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    buf136 = buf118; del buf118  # reuse
    buf137 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf139 = empty((128, ), device='cpu', dtype=torch.float32)
    buf140 = empty_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_18(c_void_p(buf129.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()))
    # Source Nodes: [out_17], Original ATen: [aten.convolution]
    buf141 = extern_kernels.convolution(buf135, buf140, primals_45, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf141, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del primals_45
    buf142 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    buf143 = buf137; del buf137  # reuse
    buf144 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf146 = empty((128, ), device='cpu', dtype=torch.float32)
    buf147 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_19(c_void_p(buf141.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    # Source Nodes: [out_18], Original ATen: [aten.convolution]
    buf148 = extern_kernels.convolution(buf142, buf147, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2)
    assert_size_stride(buf148, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del primals_48
    buf149 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    buf150 = buf144; del buf144  # reuse
    buf151 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf153 = empty((128, ), device='cpu', dtype=torch.float32)
    buf154 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_20(c_void_p(buf148.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del buf151
    # Source Nodes: [out_19], Original ATen: [aten.convolution]
    buf155 = extern_kernels.convolution(buf149, buf154, primals_51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2)
    assert_size_stride(buf155, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del primals_51
    buf156 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    buf157 = buf125; del buf125  # reuse
    buf158 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf160 = empty((512, ), device='cpu', dtype=torch.float32)
    buf161 = empty_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_21(c_void_p(buf155.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()))
    del buf158
    # Source Nodes: [out_20], Original ATen: [aten.convolution]
    buf162 = extern_kernels.convolution(buf156, buf161, primals_54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf162, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del primals_54
    buf163 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cpu', dtype=torch.float32)
    buf164 = reinterpret_tensor(buf163, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf163  # reuse
    cpp_fused_mean_22(c_void_p(buf164.data_ptr()), c_void_p(buf162.data_ptr()))
    # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
    buf165 = extern_kernels.convolution(buf164, primals_180, primals_181, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf165, (8, 128, 1, 1), (128, 1, 128, 128))
    del primals_181
    buf166 = buf165; del buf165  # reuse
    cpp_fused_relu_23(c_void_p(buf166.data_ptr()))
    # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
    buf167 = extern_kernels.convolution(buf166, primals_182, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf167, (8, 512, 1, 1), (512, 1, 512, 512))
    del primals_183
    buf168 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    buf169 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    buf170 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    buf171 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf172 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf174 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf175 = empty_strided((1536, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_avg_pool2d_mul_sigmoid_silu_view_24(c_void_p(buf162.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    # Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
    buf176 = extern_kernels.convolution(buf170, buf175, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf176, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    del primals_57
    buf177 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf178 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf180 = empty((384, ), device='cpu', dtype=torch.float32)
    buf181 = empty_strided((384, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_view_25(c_void_p(primals_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    # Source Nodes: [out_25], Original ATen: [aten.convolution]
    buf182 = extern_kernels.convolution(buf169, buf181, primals_60, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf182, (8, 384, 28, 28), (301056, 1, 10752, 384))
    del primals_60
    buf183 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    buf184 = buf178; del buf178  # reuse
    buf185 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf187 = empty((384, ), device='cpu', dtype=torch.float32)
    buf188 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_26(c_void_p(buf182.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    # Source Nodes: [out_26], Original ATen: [aten.convolution]
    buf189 = extern_kernels.convolution(buf183, buf188, primals_63, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf189, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_63
    buf190 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf191 = buf185; del buf185  # reuse
    buf192 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf194 = empty((384, ), device='cpu', dtype=torch.float32)
    buf195 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_27(c_void_p(buf189.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    # Source Nodes: [out_27], Original ATen: [aten.convolution]
    buf196 = extern_kernels.convolution(buf190, buf195, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf196, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_66
    buf197 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf198 = buf172; del buf172  # reuse
    buf199 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf201 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf202 = empty_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_28(c_void_p(buf196.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    # Source Nodes: [out_28], Original ATen: [aten.convolution]
    buf203 = extern_kernels.convolution(buf197, buf202, primals_69, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf203, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    del primals_69
    buf204 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf205 = reinterpret_tensor(buf204, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf204  # reuse
    cpp_fused_mean_29(c_void_p(buf205.data_ptr()), c_void_p(buf203.data_ptr()))
    # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
    buf206 = extern_kernels.convolution(buf205, primals_184, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf206, (8, 384, 1, 1), (384, 1, 384, 384))
    del primals_185
    buf207 = buf206; del buf206  # reuse
    cpp_fused_relu_30(c_void_p(buf207.data_ptr()))
    # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
    buf208 = extern_kernels.convolution(buf207, primals_186, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf208, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del primals_187
    buf209 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf210 = buf192; del buf192  # reuse
    buf211 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf213 = empty((384, ), device='cpu', dtype=torch.float32)
    buf214 = empty_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_31(c_void_p(buf203.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()))
    # Source Nodes: [out_33], Original ATen: [aten.convolution]
    buf215 = extern_kernels.convolution(buf209, buf214, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf215, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_72
    buf216 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf217 = buf211; del buf211  # reuse
    buf218 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf220 = empty((384, ), device='cpu', dtype=torch.float32)
    buf221 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_32(c_void_p(buf215.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    # Source Nodes: [out_34], Original ATen: [aten.convolution]
    buf222 = extern_kernels.convolution(buf216, buf221, primals_75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf222, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_75
    buf223 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf224 = buf218; del buf218  # reuse
    buf225 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf227 = empty((384, ), device='cpu', dtype=torch.float32)
    buf228 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_33(c_void_p(buf222.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    # Source Nodes: [out_35], Original ATen: [aten.convolution]
    buf229 = extern_kernels.convolution(buf223, buf228, primals_78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf229, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_78
    buf230 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf231 = buf199; del buf199  # reuse
    buf232 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf234 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf235 = empty_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_34(c_void_p(buf229.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    # Source Nodes: [out_36], Original ATen: [aten.convolution]
    buf236 = extern_kernels.convolution(buf230, buf235, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf236, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    del primals_81
    buf237 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf238 = reinterpret_tensor(buf237, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf237  # reuse
    cpp_fused_mean_35(c_void_p(buf238.data_ptr()), c_void_p(buf236.data_ptr()))
    # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
    buf239 = extern_kernels.convolution(buf238, primals_188, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf239, (8, 384, 1, 1), (384, 1, 384, 384))
    del primals_189
    buf240 = buf239; del buf239  # reuse
    cpp_fused_relu_36(c_void_p(buf240.data_ptr()))
    # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
    buf241 = extern_kernels.convolution(buf240, primals_190, primals_191, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf241, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del primals_191
    buf242 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf243 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf244 = buf225; del buf225  # reuse
    buf245 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf247 = empty((384, ), device='cpu', dtype=torch.float32)
    buf248 = empty_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_37(c_void_p(buf236.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    # Source Nodes: [out_41], Original ATen: [aten.convolution]
    buf249 = extern_kernels.convolution(buf243, buf248, primals_84, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf249, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_84
    buf250 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf251 = buf245; del buf245  # reuse
    buf252 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf254 = empty((384, ), device='cpu', dtype=torch.float32)
    buf255 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_38(c_void_p(buf249.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()))
    # Source Nodes: [out_42], Original ATen: [aten.convolution]
    buf256 = extern_kernels.convolution(buf250, buf255, primals_87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf256, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_87
    buf257 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf258 = buf252; del buf252  # reuse
    buf259 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf261 = empty((384, ), device='cpu', dtype=torch.float32)
    buf262 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_39(c_void_p(buf256.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()))
    # Source Nodes: [out_43], Original ATen: [aten.convolution]
    buf263 = extern_kernels.convolution(buf257, buf262, primals_90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf263, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_90
    buf264 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf265 = buf232; del buf232  # reuse
    buf266 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf268 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf269 = empty_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_40(c_void_p(buf263.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()))
    # Source Nodes: [out_44], Original ATen: [aten.convolution]
    buf270 = extern_kernels.convolution(buf264, buf269, primals_93, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf270, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    del primals_93
    buf271 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf272 = reinterpret_tensor(buf271, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf271  # reuse
    cpp_fused_mean_41(c_void_p(buf272.data_ptr()), c_void_p(buf270.data_ptr()))
    # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
    buf273 = extern_kernels.convolution(buf272, primals_192, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf273, (8, 384, 1, 1), (384, 1, 384, 384))
    del primals_193
    buf274 = buf273; del buf273  # reuse
    cpp_fused_relu_42(c_void_p(buf274.data_ptr()))
    # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
    buf275 = extern_kernels.convolution(buf274, primals_194, primals_195, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf275, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del primals_195
    buf276 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf277 = buf259; del buf259  # reuse
    buf278 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf280 = empty((384, ), device='cpu', dtype=torch.float32)
    buf281 = empty_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_43(c_void_p(buf270.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    # Source Nodes: [out_49], Original ATen: [aten.convolution]
    buf282 = extern_kernels.convolution(buf276, buf281, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf282, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_96
    buf283 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf284 = buf278; del buf278  # reuse
    buf285 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf287 = empty((384, ), device='cpu', dtype=torch.float32)
    buf288 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_44(c_void_p(buf282.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    # Source Nodes: [out_50], Original ATen: [aten.convolution]
    buf289 = extern_kernels.convolution(buf283, buf288, primals_99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf289, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_99
    buf290 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf291 = buf285; del buf285  # reuse
    buf292 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf294 = empty((384, ), device='cpu', dtype=torch.float32)
    buf295 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_45(c_void_p(buf289.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    # Source Nodes: [out_51], Original ATen: [aten.convolution]
    buf296 = extern_kernels.convolution(buf290, buf295, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf296, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_102
    buf297 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf298 = buf266; del buf266  # reuse
    buf299 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf301 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf302 = empty_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_46(c_void_p(buf296.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    # Source Nodes: [out_52], Original ATen: [aten.convolution]
    buf303 = extern_kernels.convolution(buf297, buf302, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf303, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    del primals_105
    buf304 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf305 = reinterpret_tensor(buf304, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf304  # reuse
    cpp_fused_mean_47(c_void_p(buf305.data_ptr()), c_void_p(buf303.data_ptr()))
    # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
    buf306 = extern_kernels.convolution(buf305, primals_196, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf306, (8, 384, 1, 1), (384, 1, 384, 384))
    del primals_197
    buf307 = buf306; del buf306  # reuse
    cpp_fused_relu_48(c_void_p(buf307.data_ptr()))
    # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
    buf308 = extern_kernels.convolution(buf307, primals_198, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf308, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del primals_199
    buf309 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf310 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf311 = buf292; del buf292  # reuse
    buf312 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf314 = empty((384, ), device='cpu', dtype=torch.float32)
    buf315 = empty_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_49(c_void_p(buf303.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()))
    # Source Nodes: [out_57], Original ATen: [aten.convolution]
    buf316 = extern_kernels.convolution(buf310, buf315, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf316, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_108
    buf317 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf318 = buf312; del buf312  # reuse
    buf319 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf321 = empty((384, ), device='cpu', dtype=torch.float32)
    buf322 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_50(c_void_p(buf316.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()))
    # Source Nodes: [out_58], Original ATen: [aten.convolution]
    buf323 = extern_kernels.convolution(buf317, buf322, primals_111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf323, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_111
    buf324 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf325 = buf319; del buf319  # reuse
    buf326 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf328 = empty((384, ), device='cpu', dtype=torch.float32)
    buf329 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_51(c_void_p(buf323.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()))
    # Source Nodes: [out_59], Original ATen: [aten.convolution]
    buf330 = extern_kernels.convolution(buf324, buf329, primals_114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf330, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_114
    buf331 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf332 = buf299; del buf299  # reuse
    buf333 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf335 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf336 = empty_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_52(c_void_p(buf330.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    # Source Nodes: [out_60], Original ATen: [aten.convolution]
    buf337 = extern_kernels.convolution(buf331, buf336, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf337, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    del primals_117
    buf338 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf339 = reinterpret_tensor(buf338, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf338  # reuse
    cpp_fused_mean_53(c_void_p(buf339.data_ptr()), c_void_p(buf337.data_ptr()))
    # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
    buf340 = extern_kernels.convolution(buf339, primals_200, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf340, (8, 384, 1, 1), (384, 1, 384, 384))
    del primals_201
    buf341 = buf340; del buf340  # reuse
    cpp_fused_relu_54(c_void_p(buf341.data_ptr()))
    # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
    buf342 = extern_kernels.convolution(buf341, primals_202, primals_203, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf342, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del primals_203
    buf343 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf344 = buf326; del buf326  # reuse
    buf345 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf347 = empty((384, ), device='cpu', dtype=torch.float32)
    buf348 = empty_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_55(c_void_p(buf337.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    # Source Nodes: [out_65], Original ATen: [aten.convolution]
    buf349 = extern_kernels.convolution(buf343, buf348, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf349, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_120
    buf350 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf351 = buf345; del buf345  # reuse
    buf352 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf354 = empty((384, ), device='cpu', dtype=torch.float32)
    buf355 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_56(c_void_p(buf349.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    # Source Nodes: [out_66], Original ATen: [aten.convolution]
    buf356 = extern_kernels.convolution(buf350, buf355, primals_123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf356, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_123
    buf357 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf358 = buf352; del buf352  # reuse
    buf359 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf361 = empty((384, ), device='cpu', dtype=torch.float32)
    buf362 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_57(c_void_p(buf356.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    # Source Nodes: [out_67], Original ATen: [aten.convolution]
    buf363 = extern_kernels.convolution(buf357, buf362, primals_126, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf363, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_126
    buf364 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf365 = buf333; del buf333  # reuse
    buf366 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf368 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf369 = empty_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_58(c_void_p(buf363.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()))
    # Source Nodes: [out_68], Original ATen: [aten.convolution]
    buf370 = extern_kernels.convolution(buf364, buf369, primals_129, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf370, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    del primals_129
    buf371 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf372 = reinterpret_tensor(buf371, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf371  # reuse
    cpp_fused_mean_59(c_void_p(buf372.data_ptr()), c_void_p(buf370.data_ptr()))
    # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
    buf373 = extern_kernels.convolution(buf372, primals_204, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf373, (8, 384, 1, 1), (384, 1, 384, 384))
    del primals_205
    buf374 = buf373; del buf373  # reuse
    cpp_fused_relu_60(c_void_p(buf374.data_ptr()))
    # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
    buf375 = extern_kernels.convolution(buf374, primals_206, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf375, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del primals_207
    buf376 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf377 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf378 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    buf379 = buf366; del buf366  # reuse
    buf380 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf382 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf383 = empty_strided((1536, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_avg_pool2d_mul_sigmoid_silu_view_61(c_void_p(buf370.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    # Source Nodes: [shortcut_13], Original ATen: [aten.convolution]
    buf384 = extern_kernels.convolution(buf378, buf383, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf384, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    del primals_132
    buf385 = buf359; del buf359  # reuse
    buf386 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf388 = empty((384, ), device='cpu', dtype=torch.float32)
    buf389 = empty_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_view_62(c_void_p(primals_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()))
    # Source Nodes: [out_73], Original ATen: [aten.convolution]
    buf390 = extern_kernels.convolution(buf377, buf389, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf390, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_135
    buf391 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf392 = buf386; del buf386  # reuse
    buf393 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf395 = empty((384, ), device='cpu', dtype=torch.float32)
    buf396 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_63(c_void_p(buf390.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    # Source Nodes: [out_74], Original ATen: [aten.convolution]
    buf397 = extern_kernels.convolution(buf391, buf396, primals_138, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf397, (8, 384, 7, 7), (18816, 1, 2688, 384))
    del primals_138
    buf398 = empty_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    buf399 = buf393; del buf393  # reuse
    buf400 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf402 = empty((384, ), device='cpu', dtype=torch.float32)
    buf403 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_64(c_void_p(buf397.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()))
    # Source Nodes: [out_75], Original ATen: [aten.convolution]
    buf404 = extern_kernels.convolution(buf398, buf403, primals_141, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf404, (8, 384, 7, 7), (18816, 1, 2688, 384))
    del primals_141
    buf405 = empty_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    buf406 = buf380; del buf380  # reuse
    buf407 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf409 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf410 = empty_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_65(c_void_p(buf404.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()))
    # Source Nodes: [out_76], Original ATen: [aten.convolution]
    buf411 = extern_kernels.convolution(buf405, buf410, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf411, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    del primals_144
    buf412 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf413 = reinterpret_tensor(buf412, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf412  # reuse
    cpp_fused_mean_66(c_void_p(buf413.data_ptr()), c_void_p(buf411.data_ptr()))
    # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
    buf414 = extern_kernels.convolution(buf413, primals_208, primals_209, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf414, (8, 384, 1, 1), (384, 1, 384, 384))
    del primals_209
    buf415 = buf414; del buf414  # reuse
    cpp_fused_relu_67(c_void_p(buf415.data_ptr()))
    # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
    buf416 = extern_kernels.convolution(buf415, primals_210, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf416, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del primals_211
    buf417 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    buf418 = buf400; del buf400  # reuse
    buf419 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf421 = empty((384, ), device='cpu', dtype=torch.float32)
    buf422 = empty_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_mul_sigmoid_silu_view_68(c_void_p(buf411.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()))
    # Source Nodes: [out_81], Original ATen: [aten.convolution]
    buf423 = extern_kernels.convolution(buf417, buf422, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf423, (8, 384, 7, 7), (18816, 1, 2688, 384))
    del primals_147
    buf424 = empty_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    buf425 = buf419; del buf419  # reuse
    buf426 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf428 = empty((384, ), device='cpu', dtype=torch.float32)
    buf429 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_69(c_void_p(buf423.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()))
    # Source Nodes: [out_82], Original ATen: [aten.convolution]
    buf430 = extern_kernels.convolution(buf424, buf429, primals_150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf430, (8, 384, 7, 7), (18816, 1, 2688, 384))
    del primals_150
    buf431 = empty_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    buf432 = buf426; del buf426  # reuse
    buf433 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf435 = empty((384, ), device='cpu', dtype=torch.float32)
    buf436 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_70(c_void_p(buf430.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    # Source Nodes: [out_83], Original ATen: [aten.convolution]
    buf437 = extern_kernels.convolution(buf431, buf436, primals_153, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf437, (8, 384, 7, 7), (18816, 1, 2688, 384))
    del primals_153
    buf438 = empty_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    buf439 = buf407; del buf407  # reuse
    buf440 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf442 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf443 = empty_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_71(c_void_p(buf437.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()))
    # Source Nodes: [out_84], Original ATen: [aten.convolution]
    buf444 = extern_kernels.convolution(buf438, buf443, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf444, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    del primals_156
    buf445 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf446 = reinterpret_tensor(buf445, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf445  # reuse
    cpp_fused_mean_72(c_void_p(buf446.data_ptr()), c_void_p(buf444.data_ptr()))
    # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
    buf447 = extern_kernels.convolution(buf446, primals_212, primals_213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf447, (8, 384, 1, 1), (384, 1, 384, 384))
    del primals_213
    buf448 = buf447; del buf447  # reuse
    cpp_fused_relu_73(c_void_p(buf448.data_ptr()))
    # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
    buf449 = extern_kernels.convolution(buf448, primals_214, primals_215, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf449, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del primals_215
    buf450 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    buf495 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    buf451 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    buf494 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    buf452 = buf433; del buf433  # reuse
    buf453 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf455 = empty((384, ), device='cpu', dtype=torch.float32)
    buf456 = empty_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_fill_mul_sigmoid_silu_sub_view_74(c_void_p(buf444.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()))
    del buf384
    # Source Nodes: [out_89], Original ATen: [aten.convolution]
    buf457 = extern_kernels.convolution(buf451, buf456, primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf457, (8, 384, 7, 7), (18816, 1, 2688, 384))
    del primals_159
    buf458 = empty_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    buf459 = buf453; del buf453  # reuse
    buf460 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf462 = empty((384, ), device='cpu', dtype=torch.float32)
    buf463 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_75(c_void_p(buf457.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()))
    # Source Nodes: [out_90], Original ATen: [aten.convolution]
    buf464 = extern_kernels.convolution(buf458, buf463, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf464, (8, 384, 7, 7), (18816, 1, 2688, 384))
    del primals_162
    buf465 = empty_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    buf466 = buf460; del buf460  # reuse
    buf467 = empty_strided((1, 384, 1), (384, 1, 384), device='cpu', dtype=torch.float32)
    buf469 = empty((384, ), device='cpu', dtype=torch.float32)
    buf470 = empty_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_76(c_void_p(buf464.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()))
    del buf467
    # Source Nodes: [out_91], Original ATen: [aten.convolution]
    buf471 = extern_kernels.convolution(buf465, buf470, primals_165, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf471, (8, 384, 7, 7), (18816, 1, 2688, 384))
    del primals_165
    buf472 = empty_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    buf473 = buf440; del buf440  # reuse
    buf474 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf476 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf477 = empty_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_silu_view_77(c_void_p(buf471.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()))
    del buf474
    # Source Nodes: [out_92], Original ATen: [aten.convolution]
    buf478 = extern_kernels.convolution(buf472, buf477, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf478, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    del primals_168
    buf479 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf480 = reinterpret_tensor(buf479, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf479  # reuse
    cpp_fused_mean_78(c_void_p(buf480.data_ptr()), c_void_p(buf478.data_ptr()))
    # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
    buf481 = extern_kernels.convolution(buf480, primals_216, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf481, (8, 384, 1, 1), (384, 1, 384, 384))
    del primals_217
    buf482 = buf481; del buf481  # reuse
    cpp_fused_relu_79(c_void_p(buf482.data_ptr()))
    # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
    buf483 = extern_kernels.convolution(buf482, primals_218, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf483, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del primals_219
    buf484 = buf450; del buf450  # reuse
    buf485 = empty_strided((1, 2304, 1), (2304, 1, 2304), device='cpu', dtype=torch.float32)
    buf486 = empty_strided((1, 2304, 1), (2304, 1, 2304), device='cpu', dtype=torch.float32)
    buf488 = empty((2304, ), device='cpu', dtype=torch.float32)
    buf489 = empty_strided((2304, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_mul_sigmoid_view_80(c_void_p(buf484.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()))
    del buf486
    # Source Nodes: [x_2], Original ATen: [aten.convolution]
    buf490 = extern_kernels.convolution(buf484, buf489, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf490, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
    del primals_171
    buf491 = empty_strided((8, 2304, 1, 1), (2304, 1, 18432, 18432), device='cpu', dtype=torch.float32)
    buf492 = reinterpret_tensor(buf491, (8, 2304), (2304, 1), 0); del buf491  # reuse
    cpp_fused_mean_silu_view_81(c_void_p(buf492.data_ptr()), c_void_p(buf490.data_ptr()))
    buf493 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_221, buf492, reinterpret_tensor(primals_220, (2304, 1000), (1, 2304), 0), alpha=1, beta=1, out=buf493)
    del primals_221
    buf496 = buf376; del buf376  # reuse
    buf497 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf498 = buf309; del buf309  # reuse
    buf499 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    buf500 = buf242; del buf242  # reuse
    buf501 = buf176; del buf176  # reuse
    buf502 = buf168; del buf168  # reuse
    buf503 = buf102; del buf102  # reuse
    buf504 = buf62; del buf62  # reuse
    cpp_fused_add_fill_mul_sigmoid_silu_sub_82(c_void_p(buf496.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()))
    return (buf493, buf0, primals_2, buf1, primals_5, buf2, primals_8, buf3, primals_11, primals_13, primals_14, primals_16, primals_17, buf4, primals_20, buf5, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, buf6, primals_35, buf7, primals_38, primals_40, primals_41, primals_43, primals_44, buf8, primals_47, buf9, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, buf10, primals_62, buf11, primals_65, primals_67, primals_68, primals_70, primals_71, buf12, primals_74, buf13, primals_77, primals_79, primals_80, primals_82, primals_83, buf14, primals_86, buf15, primals_89, primals_91, primals_92, primals_94, primals_95, buf16, primals_98, buf17, primals_101, primals_103, primals_104, primals_106, primals_107, buf18, primals_110, buf19, primals_113, primals_115, primals_116, primals_118, primals_119, buf20, primals_122, buf21, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, buf22, primals_137, buf23, primals_140, primals_142, primals_143, primals_145, primals_146, buf24, primals_149, buf25, primals_152, primals_154, primals_155, primals_157, primals_158, buf26, primals_161, buf27, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, buf28, buf32, buf33, buf34, buf35, buf39, buf40, buf41, buf42, buf46, buf47, buf48, buf49, buf53, buf54, buf55, buf56, buf60, buf61, buf66, buf67, buf68, buf69, buf73, buf74, buf75, buf76, buf80, buf81, buf82, buf83, buf87, buf88, buf89, buf91, buf93, buf94, buf95, buf96, buf100, buf101, buf106, buf107, buf108, buf109, buf113, buf114, buf115, buf116, buf120, buf121, buf122, buf123, buf127, buf128, buf129, buf131, buf133, buf134, buf135, buf139, buf140, buf141, buf142, buf146, buf147, buf148, buf149, buf153, buf154, buf155, buf156, buf160, buf161, buf162, buf164, buf166, buf167, buf169, buf170, buf174, buf175, buf180, buf181, buf182, buf183, buf187, buf188, buf189, buf190, buf194, buf195, buf196, buf197, buf201, buf202, buf203, buf205, buf207, buf208, buf209, buf213, buf214, buf215, buf216, buf220, buf221, buf222, buf223, buf227, buf228, buf229, buf230, buf234, buf235, buf236, buf238, buf240, buf241, buf243, buf247, buf248, buf249, buf250, buf254, buf255, buf256, buf257, buf261, buf262, buf263, buf264, buf268, buf269, buf270, buf272, buf274, buf275, buf276, buf280, buf281, buf282, buf283, buf287, buf288, buf289, buf290, buf294, buf295, buf296, buf297, buf301, buf302, buf303, buf305, buf307, buf308, buf310, buf314, buf315, buf316, buf317, buf321, buf322, buf323, buf324, buf328, buf329, buf330, buf331, buf335, buf336, buf337, buf339, buf341, buf342, buf343, buf347, buf348, buf349, buf350, buf354, buf355, buf356, buf357, buf361, buf362, buf363, buf364, buf368, buf369, buf370, buf372, buf374, buf375, buf377, buf378, buf382, buf383, buf388, buf389, buf390, buf391, buf395, buf396, buf397, buf398, buf402, buf403, buf404, buf405, buf409, buf410, buf411, buf413, buf415, buf416, buf417, buf421, buf422, buf423, buf424, buf428, buf429, buf430, buf431, buf435, buf436, buf437, buf438, buf442, buf443, buf444, buf446, buf448, buf449, buf451, buf455, buf456, buf457, buf458, buf462, buf463, buf464, buf465, buf469, buf470, buf471, buf472, buf476, buf477, buf478, buf480, buf482, buf483, buf484, buf488, buf489, buf490, buf492, reinterpret_tensor(primals_220, (1000, 2304), (2304, 1), 0), reinterpret_tensor(buf485, (1, 2304, 1), (2304, 1, 1), 0), reinterpret_tensor(buf473, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf466, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf459, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf452, (1, 384, 1), (384, 1, 1), 0), buf494, reinterpret_tensor(buf439, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf432, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf425, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf418, (1, 384, 1), (384, 1, 1), 0), buf495, reinterpret_tensor(buf406, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf399, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf392, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf385, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf379, (1, 1536, 1), (1536, 1, 1), 0), buf496, reinterpret_tensor(buf365, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf358, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf351, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf344, (1, 384, 1), (384, 1, 1), 0), buf497, reinterpret_tensor(buf332, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf325, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf318, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf311, (1, 384, 1), (384, 1, 1), 0), buf498, reinterpret_tensor(buf298, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf291, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf284, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf277, (1, 384, 1), (384, 1, 1), 0), buf499, reinterpret_tensor(buf265, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf258, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf251, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf244, (1, 384, 1), (384, 1, 1), 0), buf500, reinterpret_tensor(buf231, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf224, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf217, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf210, (1, 384, 1), (384, 1, 1), 0), buf501, reinterpret_tensor(buf198, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf191, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf184, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf177, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf171, (1, 1536, 1), (1536, 1, 1), 0), buf502, reinterpret_tensor(buf157, (1, 512, 1), (512, 1, 1), 0), reinterpret_tensor(buf150, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf143, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf136, (1, 128, 1), (128, 1, 1), 0), buf503, reinterpret_tensor(buf124, (1, 512, 1), (512, 1, 1), 0), reinterpret_tensor(buf117, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf110, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf103, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf97, (1, 512, 1), (512, 1, 1), 0), buf504, reinterpret_tensor(buf84, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf77, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf70, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf63, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf57, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf50, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf43, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf36, (1, 32, 1), (32, 1, 1), 0), reinterpret_tensor(buf29, (1, 16, 1), (16, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((384, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((2304, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((2304, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((1000, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nfnet_l0', benchmark_compiled_module)
