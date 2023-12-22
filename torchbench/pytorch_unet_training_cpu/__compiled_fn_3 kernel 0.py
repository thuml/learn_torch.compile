
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
                       float* out_ptr18)
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr8[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr9 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr9[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (9216L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr10 + static_cast<long>(x1 + (1024L*x2) + (9216L*x0)), static_cast<long>(1024L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr10[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (9216L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (1024L*x2) + (9216L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr11 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr11[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr12 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr12[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr13 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr13[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr14 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr14[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr15 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr15[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr15 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr16 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr16[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr16 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                        }
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(613760L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr18[static_cast<long>(x2 + (613760L*x1) + (1841280L*x0))];
                        out_ptr18[static_cast<long>(x1 + (3L*x2) + (1841280L*x0))] = tmp0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(479L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (122752L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(64L + x2 + (128L*x1) + (122752L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(61376L + x2 + (128L*x1) + (122752L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(61440L + x2 + (128L*x1) + (122752L*x0)));
                        auto tmp2 = at::vec::maximum(tmp1, tmp0);
                        auto tmp4 = at::vec::maximum(tmp3, tmp2);
                        auto tmp6 = at::vec::maximum(tmp5, tmp4);
                        tmp6.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (30656L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(479L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(x3 + (128L*x2) + (122752L*x1) + (39280640L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(64L + x3 + (128L*x2) + (122752L*x1) + (39280640L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(61376L + x3 + (128L*x2) + (122752L*x1) + (39280640L*x0))];
                            auto tmp12 = out_ptr0[static_cast<long>(61440L + x3 + (128L*x2) + (122752L*x1) + (39280640L*x0))];
                            auto tmp2 = tmp1 > tmp0;
                            auto tmp3 = c10::convert<long>(1L + (2L*x2) + (1918L*x1));
                            auto tmp4 = c10::convert<long>((2L*x2) + (1918L*x1));
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp6 = max_propagate_nan(tmp1, tmp0);
                            auto tmp8 = tmp7 > tmp6;
                            auto tmp9 = c10::convert<long>(959L + (2L*x2) + (1918L*x1));
                            auto tmp10 = tmp8 ? tmp9 : tmp5;
                            auto tmp11 = max_propagate_nan(tmp7, tmp6);
                            auto tmp13 = tmp12 > tmp11;
                            auto tmp14 = c10::convert<long>(960L + (2L*x2) + (1918L*x1));
                            auto tmp15 = tmp13 ? tmp14 : tmp10;
                            auto tmp16 = max_propagate_nan(tmp12, tmp11);
                            out_ptr2[static_cast<long>(x3 + (64L*x2) + (30656L*x1) + (9809920L*x0))] = tmp15;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(306560L); x0+=static_cast<long>(1L))
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
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(306560L); x0+=static_cast<long>(1L))
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
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(239L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (122624L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(128L + x2 + (256L*x1) + (122624L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(61312L + x2 + (256L*x1) + (122624L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(61440L + x2 + (256L*x1) + (122624L*x0)));
                        auto tmp2 = at::vec::maximum(tmp1, tmp0);
                        auto tmp4 = at::vec::maximum(tmp3, tmp2);
                        auto tmp6 = at::vec::maximum(tmp5, tmp4);
                        tmp6.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (30592L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(239L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(x3 + (256L*x2) + (122624L*x1) + (19619840L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(128L + x3 + (256L*x2) + (122624L*x1) + (19619840L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(61312L + x3 + (256L*x2) + (122624L*x1) + (19619840L*x0))];
                            auto tmp12 = out_ptr0[static_cast<long>(61440L + x3 + (256L*x2) + (122624L*x1) + (19619840L*x0))];
                            auto tmp2 = tmp1 > tmp0;
                            auto tmp3 = c10::convert<long>(1L + (2L*x2) + (958L*x1));
                            auto tmp4 = c10::convert<long>((2L*x2) + (958L*x1));
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp6 = max_propagate_nan(tmp1, tmp0);
                            auto tmp8 = tmp7 > tmp6;
                            auto tmp9 = c10::convert<long>(479L + (2L*x2) + (958L*x1));
                            auto tmp10 = tmp8 ? tmp9 : tmp5;
                            auto tmp11 = max_propagate_nan(tmp7, tmp6);
                            auto tmp13 = tmp12 > tmp11;
                            auto tmp14 = c10::convert<long>(480L + (2L*x2) + (958L*x1));
                            auto tmp15 = tmp13 ? tmp14 : tmp10;
                            auto tmp16 = max_propagate_nan(tmp12, tmp11);
                            out_ptr2[static_cast<long>(x3 + (128L*x2) + (30592L*x1) + (4894720L*x0))] = tmp15;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(76480L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(76480L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(119L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (122368L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x2 + (512L*x1) + (122368L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(61184L + x2 + (512L*x1) + (122368L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(61440L + x2 + (512L*x1) + (122368L*x0)));
                        auto tmp2 = at::vec::maximum(tmp1, tmp0);
                        auto tmp4 = at::vec::maximum(tmp3, tmp2);
                        auto tmp6 = at::vec::maximum(tmp5, tmp4);
                        tmp6.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (30464L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(119L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(x3 + (512L*x2) + (122368L*x1) + (9789440L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(256L + x3 + (512L*x2) + (122368L*x1) + (9789440L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(61184L + x3 + (512L*x2) + (122368L*x1) + (9789440L*x0))];
                            auto tmp12 = out_ptr0[static_cast<long>(61440L + x3 + (512L*x2) + (122368L*x1) + (9789440L*x0))];
                            auto tmp2 = tmp1 > tmp0;
                            auto tmp3 = c10::convert<long>(1L + (2L*x2) + (478L*x1));
                            auto tmp4 = c10::convert<long>((2L*x2) + (478L*x1));
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp6 = max_propagate_nan(tmp1, tmp0);
                            auto tmp8 = tmp7 > tmp6;
                            auto tmp9 = c10::convert<long>(239L + (2L*x2) + (478L*x1));
                            auto tmp10 = tmp8 ? tmp9 : tmp5;
                            auto tmp11 = max_propagate_nan(tmp7, tmp6);
                            auto tmp13 = tmp12 > tmp11;
                            auto tmp14 = c10::convert<long>(240L + (2L*x2) + (478L*x1));
                            auto tmp15 = tmp13 ? tmp14 : tmp10;
                            auto tmp16 = max_propagate_nan(tmp12, tmp11);
                            out_ptr2[static_cast<long>(x3 + (256L*x2) + (30464L*x1) + (2437120L*x0))] = tmp15;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19040L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19040L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(59L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (121856L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (121856L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(60928L + x2 + (1024L*x1) + (121856L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(61440L + x2 + (1024L*x1) + (121856L*x0)));
                        auto tmp2 = at::vec::maximum(tmp1, tmp0);
                        auto tmp4 = at::vec::maximum(tmp3, tmp2);
                        auto tmp6 = at::vec::maximum(tmp5, tmp4);
                        tmp6.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (30208L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(59L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(x3 + (1024L*x2) + (121856L*x1) + (4874240L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(512L + x3 + (1024L*x2) + (121856L*x1) + (4874240L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(60928L + x3 + (1024L*x2) + (121856L*x1) + (4874240L*x0))];
                            auto tmp12 = out_ptr0[static_cast<long>(61440L + x3 + (1024L*x2) + (121856L*x1) + (4874240L*x0))];
                            auto tmp2 = tmp1 > tmp0;
                            auto tmp3 = c10::convert<long>(1L + (2L*x2) + (238L*x1));
                            auto tmp4 = c10::convert<long>((2L*x2) + (238L*x1));
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp6 = max_propagate_nan(tmp1, tmp0);
                            auto tmp8 = tmp7 > tmp6;
                            auto tmp9 = c10::convert<long>(119L + (2L*x2) + (238L*x1));
                            auto tmp10 = tmp8 ? tmp9 : tmp5;
                            auto tmp11 = max_propagate_nan(tmp7, tmp6);
                            auto tmp13 = tmp12 > tmp11;
                            auto tmp14 = c10::convert<long>(120L + (2L*x2) + (238L*x1));
                            auto tmp15 = tmp13 ? tmp14 : tmp10;
                            auto tmp16 = max_propagate_nan(tmp12, tmp11);
                            out_ptr2[static_cast<long>(x3 + (512L*x2) + (30208L*x1) + (1208320L*x0))] = tmp15;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4720L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_cat_ceil_clamp_mul_relu_rsub_sub_unsqueeze_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       long* out_ptr1,
                       long* out_ptr2,
                       long* out_ptr3,
                       long* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4720L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(118L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.49572649572649574);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    out_ptr1[static_cast<long>(x0)] = tmp9;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(118L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.49572649572649574);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = static_cast<float>(58.0);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    out_ptr2[static_cast<long>(x0)] = tmp12;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4936708860759494);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    out_ptr3[static_cast<long>(x0)] = tmp9;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4936708860759494);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = static_cast<float>(39.0);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    out_ptr4[static_cast<long>(x0)] = tmp12;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4936708860759494);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<float>(tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    out_ptr5[static_cast<long>(x0)] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(118L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.49572649572649574);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<float>(tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    out_ptr6[static_cast<long>(x0)] = tmp11;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(118L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x2)];
                            auto tmp4 = out_ptr1[static_cast<long>(x3)];
                            auto tmp24 = out_ptr4[static_cast<long>(x2)];
                            auto tmp29 = out_ptr5[static_cast<long>(x2)];
                            auto tmp1 = decltype(tmp0)(tmp0 + 40);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            auto tmp5 = decltype(tmp4)(tmp4 + 59);
                            auto tmp6 = tmp4 < 0;
                            auto tmp7 = tmp6 ? tmp5 : tmp4;
                            auto tmp8 = out_ptr0[static_cast<long>(x1 + (512L*tmp7) + (30208L*tmp3) + (1208320L*x0))];
                            auto tmp9 = c10::convert<long>(x2);
                            auto tmp10 = c10::convert<double>(tmp9);
                            auto tmp11 = static_cast<double>(1.0);
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = static_cast<double>(0.0);
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = c10::convert<float>(tmp14);
                            auto tmp16 = static_cast<float>(0.4936708860759494);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = c10::convert<long>(tmp17);
                            auto tmp19 = c10::convert<float>(tmp18);
                            auto tmp20 = decltype(tmp17)(tmp17 - tmp19);
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                            auto tmp23 = decltype(tmp8)(tmp8 * tmp22);
                            auto tmp25 = decltype(tmp24)(tmp24 + 40);
                            auto tmp26 = tmp24 < 0;
                            auto tmp27 = tmp26 ? tmp25 : tmp24;
                            auto tmp28 = out_ptr0[static_cast<long>(x1 + (512L*tmp7) + (30208L*tmp27) + (1208320L*x0))];
                            auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                            auto tmp31 = decltype(tmp23)(tmp23 + tmp30);
                            auto tmp32 = c10::convert<long>(x3);
                            auto tmp33 = c10::convert<double>(tmp32);
                            auto tmp34 = decltype(tmp33)(tmp33 * tmp11);
                            auto tmp35 = decltype(tmp34)(tmp34 + tmp13);
                            auto tmp36 = c10::convert<float>(tmp35);
                            auto tmp37 = static_cast<float>(0.49572649572649574);
                            auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                            auto tmp39 = c10::convert<long>(tmp38);
                            auto tmp40 = c10::convert<float>(tmp39);
                            auto tmp41 = decltype(tmp38)(tmp38 - tmp40);
                            auto tmp42 = decltype(tmp21)(tmp21 - tmp41);
                            auto tmp43 = decltype(tmp31)(tmp31 * tmp42);
                            out_ptr7[static_cast<long>(x3 + (118L*x2) + (9440L*x1) + (4833280L*x0))] = tmp43;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(119L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(512);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (512L*x2) + (60928L*x1) + (4874240L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(1024);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(118);
                                auto tmp14 = tmp12 < tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr7[static_cast<long>((-4833280L) + x2 + (118L*x1) + (9440L*x3) + (4833280L*x0))];
                                    auto tmp17 = out_ptr3[static_cast<long>(x1)];
                                    auto tmp18 = decltype(tmp17)(tmp17 + 40);
                                    auto tmp19 = tmp17 < 0;
                                    auto tmp20 = tmp19 ? tmp18 : tmp17;
                                    auto tmp21 = out_ptr2[static_cast<long>(x2)];
                                    auto tmp22 = decltype(tmp21)(tmp21 + 59);
                                    auto tmp23 = tmp21 < 0;
                                    auto tmp24 = tmp23 ? tmp22 : tmp21;
                                    auto tmp25 = out_ptr0[static_cast<long>((-512L) + x3 + (512L*tmp24) + (30208L*tmp20) + (1208320L*x0))];
                                    auto tmp26 = c10::convert<long>(x1);
                                    auto tmp27 = c10::convert<double>(tmp26);
                                    auto tmp28 = static_cast<double>(1.0);
                                    auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                                    auto tmp30 = static_cast<double>(0.0);
                                    auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                                    auto tmp32 = c10::convert<float>(tmp31);
                                    auto tmp33 = static_cast<float>(0.4936708860759494);
                                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                                    auto tmp35 = c10::convert<long>(tmp34);
                                    auto tmp36 = c10::convert<float>(tmp35);
                                    auto tmp37 = decltype(tmp34)(tmp34 - tmp36);
                                    auto tmp38 = static_cast<float>(1.0);
                                    auto tmp39 = decltype(tmp38)(tmp38 - tmp37);
                                    auto tmp40 = decltype(tmp25)(tmp25 * tmp39);
                                    auto tmp41 = out_ptr4[static_cast<long>(x1)];
                                    auto tmp42 = decltype(tmp41)(tmp41 + 40);
                                    auto tmp43 = tmp41 < 0;
                                    auto tmp44 = tmp43 ? tmp42 : tmp41;
                                    auto tmp45 = out_ptr0[static_cast<long>((-512L) + x3 + (512L*tmp24) + (30208L*tmp44) + (1208320L*x0))];
                                    auto tmp46 = out_ptr5[static_cast<long>(x1)];
                                    auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                                    auto tmp48 = decltype(tmp40)(tmp40 + tmp47);
                                    auto tmp49 = out_ptr6[static_cast<long>(x2)];
                                    auto tmp50 = decltype(tmp48)(tmp48 * tmp49);
                                    auto tmp51 = decltype(tmp16)(tmp16 + tmp50);
                                    return tmp51;
                                }
                                ;
                                auto tmp52 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                return tmp52;
                            }
                            ;
                            auto tmp53 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp54 = tmp4 ? tmp7 : tmp53;
                            out_ptr8[static_cast<long>(x3 + (1024L*x2) + (121856L*x1) + (9748480L*x0))] = tmp54;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19040L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_cat_ceil_clamp_mul_relu_rsub_sub_unsqueeze_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       long* out_ptr1,
                       long* out_ptr2,
                       long* out_ptr3,
                       long* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19040L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(238L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4978902953586498);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    out_ptr1[static_cast<long>(x0)] = tmp9;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(238L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4978902953586498);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = static_cast<float>(118.0);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    out_ptr2[static_cast<long>(x0)] = tmp12;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4968553459119497);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    out_ptr3[static_cast<long>(x0)] = tmp9;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4968553459119497);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = static_cast<float>(79.0);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    out_ptr4[static_cast<long>(x0)] = tmp12;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4968553459119497);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<float>(tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    out_ptr5[static_cast<long>(x0)] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(238L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4978902953586498);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<float>(tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    out_ptr6[static_cast<long>(x0)] = tmp11;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(160L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(238L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x2)];
                            auto tmp4 = out_ptr1[static_cast<long>(x3)];
                            auto tmp24 = out_ptr4[static_cast<long>(x2)];
                            auto tmp29 = out_ptr5[static_cast<long>(x2)];
                            auto tmp1 = decltype(tmp0)(tmp0 + 80);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            auto tmp5 = decltype(tmp4)(tmp4 + 119);
                            auto tmp6 = tmp4 < 0;
                            auto tmp7 = tmp6 ? tmp5 : tmp4;
                            auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*tmp7) + (30464L*tmp3) + (2437120L*x0))];
                            auto tmp9 = c10::convert<long>(x2);
                            auto tmp10 = c10::convert<double>(tmp9);
                            auto tmp11 = static_cast<double>(1.0);
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = static_cast<double>(0.0);
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = c10::convert<float>(tmp14);
                            auto tmp16 = static_cast<float>(0.4968553459119497);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = c10::convert<long>(tmp17);
                            auto tmp19 = c10::convert<float>(tmp18);
                            auto tmp20 = decltype(tmp17)(tmp17 - tmp19);
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                            auto tmp23 = decltype(tmp8)(tmp8 * tmp22);
                            auto tmp25 = decltype(tmp24)(tmp24 + 80);
                            auto tmp26 = tmp24 < 0;
                            auto tmp27 = tmp26 ? tmp25 : tmp24;
                            auto tmp28 = out_ptr0[static_cast<long>(x1 + (256L*tmp7) + (30464L*tmp27) + (2437120L*x0))];
                            auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                            auto tmp31 = decltype(tmp23)(tmp23 + tmp30);
                            auto tmp32 = c10::convert<long>(x3);
                            auto tmp33 = c10::convert<double>(tmp32);
                            auto tmp34 = decltype(tmp33)(tmp33 * tmp11);
                            auto tmp35 = decltype(tmp34)(tmp34 + tmp13);
                            auto tmp36 = c10::convert<float>(tmp35);
                            auto tmp37 = static_cast<float>(0.4978902953586498);
                            auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                            auto tmp39 = c10::convert<long>(tmp38);
                            auto tmp40 = c10::convert<float>(tmp39);
                            auto tmp41 = decltype(tmp38)(tmp38 - tmp40);
                            auto tmp42 = decltype(tmp21)(tmp21 - tmp41);
                            auto tmp43 = decltype(tmp31)(tmp31 * tmp42);
                            out_ptr7[static_cast<long>(x3 + (238L*x2) + (38080L*x1) + (9748480L*x0))] = tmp43;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(239L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(256);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (256L*x2) + (61184L*x1) + (9789440L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(512);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(238);
                                auto tmp14 = tmp12 < tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr7[static_cast<long>((-9748480L) + x2 + (238L*x1) + (38080L*x3) + (9748480L*x0))];
                                    auto tmp17 = out_ptr3[static_cast<long>(x1)];
                                    auto tmp18 = decltype(tmp17)(tmp17 + 80);
                                    auto tmp19 = tmp17 < 0;
                                    auto tmp20 = tmp19 ? tmp18 : tmp17;
                                    auto tmp21 = out_ptr2[static_cast<long>(x2)];
                                    auto tmp22 = decltype(tmp21)(tmp21 + 119);
                                    auto tmp23 = tmp21 < 0;
                                    auto tmp24 = tmp23 ? tmp22 : tmp21;
                                    auto tmp25 = out_ptr0[static_cast<long>((-256L) + x3 + (256L*tmp24) + (30464L*tmp20) + (2437120L*x0))];
                                    auto tmp26 = c10::convert<long>(x1);
                                    auto tmp27 = c10::convert<double>(tmp26);
                                    auto tmp28 = static_cast<double>(1.0);
                                    auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                                    auto tmp30 = static_cast<double>(0.0);
                                    auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                                    auto tmp32 = c10::convert<float>(tmp31);
                                    auto tmp33 = static_cast<float>(0.4968553459119497);
                                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                                    auto tmp35 = c10::convert<long>(tmp34);
                                    auto tmp36 = c10::convert<float>(tmp35);
                                    auto tmp37 = decltype(tmp34)(tmp34 - tmp36);
                                    auto tmp38 = static_cast<float>(1.0);
                                    auto tmp39 = decltype(tmp38)(tmp38 - tmp37);
                                    auto tmp40 = decltype(tmp25)(tmp25 * tmp39);
                                    auto tmp41 = out_ptr4[static_cast<long>(x1)];
                                    auto tmp42 = decltype(tmp41)(tmp41 + 80);
                                    auto tmp43 = tmp41 < 0;
                                    auto tmp44 = tmp43 ? tmp42 : tmp41;
                                    auto tmp45 = out_ptr0[static_cast<long>((-256L) + x3 + (256L*tmp24) + (30464L*tmp44) + (2437120L*x0))];
                                    auto tmp46 = out_ptr5[static_cast<long>(x1)];
                                    auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                                    auto tmp48 = decltype(tmp40)(tmp40 + tmp47);
                                    auto tmp49 = out_ptr6[static_cast<long>(x2)];
                                    auto tmp50 = decltype(tmp48)(tmp48 * tmp49);
                                    auto tmp51 = decltype(tmp16)(tmp16 + tmp50);
                                    return tmp51;
                                }
                                ;
                                auto tmp52 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                return tmp52;
                            }
                            ;
                            auto tmp53 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp54 = tmp4 ? tmp7 : tmp53;
                            out_ptr8[static_cast<long>(x3 + (512L*x2) + (122368L*x1) + (19578880L*x0))] = tmp54;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(76480L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_cat_ceil_clamp_mul_relu_rsub_sub_unsqueeze_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       long* out_ptr1,
                       long* out_ptr2,
                       long* out_ptr3,
                       long* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(76480L); x0+=static_cast<long>(1L))
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
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(478L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4989517819706499);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    out_ptr1[static_cast<long>(x0)] = tmp9;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(478L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4989517819706499);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = static_cast<float>(238.0);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    out_ptr2[static_cast<long>(x0)] = tmp12;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.49843260188087773);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    out_ptr3[static_cast<long>(x0)] = tmp9;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.49843260188087773);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = static_cast<float>(159.0);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    out_ptr4[static_cast<long>(x0)] = tmp12;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.49843260188087773);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<float>(tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    out_ptr5[static_cast<long>(x0)] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(478L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4989517819706499);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<float>(tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    out_ptr6[static_cast<long>(x0)] = tmp11;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(478L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x2)];
                            auto tmp4 = out_ptr1[static_cast<long>(x3)];
                            auto tmp24 = out_ptr4[static_cast<long>(x2)];
                            auto tmp29 = out_ptr5[static_cast<long>(x2)];
                            auto tmp1 = decltype(tmp0)(tmp0 + 160);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            auto tmp5 = decltype(tmp4)(tmp4 + 239);
                            auto tmp6 = tmp4 < 0;
                            auto tmp7 = tmp6 ? tmp5 : tmp4;
                            auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*tmp7) + (30592L*tmp3) + (4894720L*x0))];
                            auto tmp9 = c10::convert<long>(x2);
                            auto tmp10 = c10::convert<double>(tmp9);
                            auto tmp11 = static_cast<double>(1.0);
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = static_cast<double>(0.0);
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = c10::convert<float>(tmp14);
                            auto tmp16 = static_cast<float>(0.49843260188087773);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = c10::convert<long>(tmp17);
                            auto tmp19 = c10::convert<float>(tmp18);
                            auto tmp20 = decltype(tmp17)(tmp17 - tmp19);
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                            auto tmp23 = decltype(tmp8)(tmp8 * tmp22);
                            auto tmp25 = decltype(tmp24)(tmp24 + 160);
                            auto tmp26 = tmp24 < 0;
                            auto tmp27 = tmp26 ? tmp25 : tmp24;
                            auto tmp28 = out_ptr0[static_cast<long>(x1 + (128L*tmp7) + (30592L*tmp27) + (4894720L*x0))];
                            auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                            auto tmp31 = decltype(tmp23)(tmp23 + tmp30);
                            auto tmp32 = c10::convert<long>(x3);
                            auto tmp33 = c10::convert<double>(tmp32);
                            auto tmp34 = decltype(tmp33)(tmp33 * tmp11);
                            auto tmp35 = decltype(tmp34)(tmp34 + tmp13);
                            auto tmp36 = c10::convert<float>(tmp35);
                            auto tmp37 = static_cast<float>(0.4989517819706499);
                            auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                            auto tmp39 = c10::convert<long>(tmp38);
                            auto tmp40 = c10::convert<float>(tmp39);
                            auto tmp41 = decltype(tmp38)(tmp38 - tmp40);
                            auto tmp42 = decltype(tmp21)(tmp21 - tmp41);
                            auto tmp43 = decltype(tmp31)(tmp31 * tmp42);
                            out_ptr7[static_cast<long>(x3 + (478L*x2) + (152960L*x1) + (19578880L*x0))] = tmp43;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(479L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(128);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (128L*x2) + (61312L*x1) + (19619840L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(256);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(478);
                                auto tmp14 = tmp12 < tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr7[static_cast<long>((-19578880L) + x2 + (478L*x1) + (152960L*x3) + (19578880L*x0))];
                                    auto tmp17 = out_ptr3[static_cast<long>(x1)];
                                    auto tmp18 = decltype(tmp17)(tmp17 + 160);
                                    auto tmp19 = tmp17 < 0;
                                    auto tmp20 = tmp19 ? tmp18 : tmp17;
                                    auto tmp21 = out_ptr2[static_cast<long>(x2)];
                                    auto tmp22 = decltype(tmp21)(tmp21 + 239);
                                    auto tmp23 = tmp21 < 0;
                                    auto tmp24 = tmp23 ? tmp22 : tmp21;
                                    auto tmp25 = out_ptr0[static_cast<long>((-128L) + x3 + (128L*tmp24) + (30592L*tmp20) + (4894720L*x0))];
                                    auto tmp26 = c10::convert<long>(x1);
                                    auto tmp27 = c10::convert<double>(tmp26);
                                    auto tmp28 = static_cast<double>(1.0);
                                    auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                                    auto tmp30 = static_cast<double>(0.0);
                                    auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                                    auto tmp32 = c10::convert<float>(tmp31);
                                    auto tmp33 = static_cast<float>(0.49843260188087773);
                                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                                    auto tmp35 = c10::convert<long>(tmp34);
                                    auto tmp36 = c10::convert<float>(tmp35);
                                    auto tmp37 = decltype(tmp34)(tmp34 - tmp36);
                                    auto tmp38 = static_cast<float>(1.0);
                                    auto tmp39 = decltype(tmp38)(tmp38 - tmp37);
                                    auto tmp40 = decltype(tmp25)(tmp25 * tmp39);
                                    auto tmp41 = out_ptr4[static_cast<long>(x1)];
                                    auto tmp42 = decltype(tmp41)(tmp41 + 160);
                                    auto tmp43 = tmp41 < 0;
                                    auto tmp44 = tmp43 ? tmp42 : tmp41;
                                    auto tmp45 = out_ptr0[static_cast<long>((-128L) + x3 + (128L*tmp24) + (30592L*tmp44) + (4894720L*x0))];
                                    auto tmp46 = out_ptr5[static_cast<long>(x1)];
                                    auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                                    auto tmp48 = decltype(tmp40)(tmp40 + tmp47);
                                    auto tmp49 = out_ptr6[static_cast<long>(x2)];
                                    auto tmp50 = decltype(tmp48)(tmp48 * tmp49);
                                    auto tmp51 = decltype(tmp16)(tmp16 + tmp50);
                                    return tmp51;
                                }
                                ;
                                auto tmp52 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                return tmp52;
                            }
                            ;
                            auto tmp53 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp54 = tmp4 ? tmp7 : tmp53;
                            out_ptr8[static_cast<long>(x3 + (256L*x2) + (122624L*x1) + (39239680L*x0))] = tmp54;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(306560L); x0+=static_cast<long>(1L))
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
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_cat_ceil_clamp_mul_relu_rsub_sub_unsqueeze_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       long* out_ptr1,
                       long* out_ptr2,
                       long* out_ptr3,
                       long* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(306560L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(958L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4994775339602926);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    out_ptr1[static_cast<long>(x0)] = tmp9;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(958L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4994775339602926);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = static_cast<float>(478.0);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    out_ptr2[static_cast<long>(x0)] = tmp12;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.49921752738654146);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    out_ptr3[static_cast<long>(x0)] = tmp9;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.49921752738654146);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = static_cast<float>(319.0);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    out_ptr4[static_cast<long>(x0)] = tmp12;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.49921752738654146);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<float>(tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    out_ptr5[static_cast<long>(x0)] = tmp11;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(958L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.4994775339602926);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<float>(tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    out_ptr6[static_cast<long>(x0)] = tmp11;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(640L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(958L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x2)];
                            auto tmp4 = out_ptr1[static_cast<long>(x3)];
                            auto tmp24 = out_ptr4[static_cast<long>(x2)];
                            auto tmp29 = out_ptr5[static_cast<long>(x2)];
                            auto tmp1 = decltype(tmp0)(tmp0 + 320);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            auto tmp5 = decltype(tmp4)(tmp4 + 479);
                            auto tmp6 = tmp4 < 0;
                            auto tmp7 = tmp6 ? tmp5 : tmp4;
                            auto tmp8 = out_ptr0[static_cast<long>(x1 + (64L*tmp7) + (30656L*tmp3) + (9809920L*x0))];
                            auto tmp9 = c10::convert<long>(x2);
                            auto tmp10 = c10::convert<double>(tmp9);
                            auto tmp11 = static_cast<double>(1.0);
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = static_cast<double>(0.0);
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = c10::convert<float>(tmp14);
                            auto tmp16 = static_cast<float>(0.49921752738654146);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = c10::convert<long>(tmp17);
                            auto tmp19 = c10::convert<float>(tmp18);
                            auto tmp20 = decltype(tmp17)(tmp17 - tmp19);
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                            auto tmp23 = decltype(tmp8)(tmp8 * tmp22);
                            auto tmp25 = decltype(tmp24)(tmp24 + 320);
                            auto tmp26 = tmp24 < 0;
                            auto tmp27 = tmp26 ? tmp25 : tmp24;
                            auto tmp28 = out_ptr0[static_cast<long>(x1 + (64L*tmp7) + (30656L*tmp27) + (9809920L*x0))];
                            auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                            auto tmp31 = decltype(tmp23)(tmp23 + tmp30);
                            auto tmp32 = c10::convert<long>(x3);
                            auto tmp33 = c10::convert<double>(tmp32);
                            auto tmp34 = decltype(tmp33)(tmp33 * tmp11);
                            auto tmp35 = decltype(tmp34)(tmp34 + tmp13);
                            auto tmp36 = c10::convert<float>(tmp35);
                            auto tmp37 = static_cast<float>(0.4994775339602926);
                            auto tmp38 = decltype(tmp36)(tmp36 * tmp37);
                            auto tmp39 = c10::convert<long>(tmp38);
                            auto tmp40 = c10::convert<float>(tmp39);
                            auto tmp41 = decltype(tmp38)(tmp38 - tmp40);
                            auto tmp42 = decltype(tmp21)(tmp21 - tmp41);
                            auto tmp43 = decltype(tmp31)(tmp31 * tmp42);
                            out_ptr7[static_cast<long>(x3 + (958L*x2) + (613120L*x1) + (39239680L*x0))] = tmp43;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(959L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(64);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr5[static_cast<long>(x3 + (64L*x2) + (61376L*x1) + (39280640L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(128);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(x2);
                                auto tmp13 = static_cast<long>(958);
                                auto tmp14 = tmp12 < tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr7[static_cast<long>((-39239680L) + x2 + (958L*x1) + (613120L*x3) + (39239680L*x0))];
                                    auto tmp17 = out_ptr3[static_cast<long>(x1)];
                                    auto tmp18 = decltype(tmp17)(tmp17 + 320);
                                    auto tmp19 = tmp17 < 0;
                                    auto tmp20 = tmp19 ? tmp18 : tmp17;
                                    auto tmp21 = out_ptr2[static_cast<long>(x2)];
                                    auto tmp22 = decltype(tmp21)(tmp21 + 479);
                                    auto tmp23 = tmp21 < 0;
                                    auto tmp24 = tmp23 ? tmp22 : tmp21;
                                    auto tmp25 = out_ptr0[static_cast<long>((-64L) + x3 + (64L*tmp24) + (30656L*tmp20) + (9809920L*x0))];
                                    auto tmp26 = c10::convert<long>(x1);
                                    auto tmp27 = c10::convert<double>(tmp26);
                                    auto tmp28 = static_cast<double>(1.0);
                                    auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                                    auto tmp30 = static_cast<double>(0.0);
                                    auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                                    auto tmp32 = c10::convert<float>(tmp31);
                                    auto tmp33 = static_cast<float>(0.49921752738654146);
                                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                                    auto tmp35 = c10::convert<long>(tmp34);
                                    auto tmp36 = c10::convert<float>(tmp35);
                                    auto tmp37 = decltype(tmp34)(tmp34 - tmp36);
                                    auto tmp38 = static_cast<float>(1.0);
                                    auto tmp39 = decltype(tmp38)(tmp38 - tmp37);
                                    auto tmp40 = decltype(tmp25)(tmp25 * tmp39);
                                    auto tmp41 = out_ptr4[static_cast<long>(x1)];
                                    auto tmp42 = decltype(tmp41)(tmp41 + 320);
                                    auto tmp43 = tmp41 < 0;
                                    auto tmp44 = tmp43 ? tmp42 : tmp41;
                                    auto tmp45 = out_ptr0[static_cast<long>((-64L) + x3 + (64L*tmp24) + (30656L*tmp44) + (9809920L*x0))];
                                    auto tmp46 = out_ptr5[static_cast<long>(x1)];
                                    auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                                    auto tmp48 = decltype(tmp40)(tmp40 + tmp47);
                                    auto tmp49 = out_ptr6[static_cast<long>(x2)];
                                    auto tmp50 = decltype(tmp48)(tmp48 * tmp49);
                                    auto tmp51 = decltype(tmp16)(tmp16 + tmp50);
                                    return tmp51;
                                }
                                ;
                                auto tmp52 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                return tmp52;
                            }
                            ;
                            auto tmp53 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp54 = tmp4 ? tmp7 : tmp53;
                            out_ptr8[static_cast<long>(x3 + (128L*x2) + (122752L*x1) + (78561280L*x0))] = tmp54;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
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


cpp_fused_convolution_threshold_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       bool* out_ptr1,
                       bool* out_ptr2,
                       bool* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(613760L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (2L*x2) + (1227520L*x0))];
                        out_ptr0[static_cast<long>(x2 + (613760L*x1) + (1227520L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19619840L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr1[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr1[static_cast<long>(x0)] = tmp2;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9789440L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr2[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr2[static_cast<long>(x0)] = tmp2;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4874240L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr3[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr3[static_cast<long>(x0)] = tmp2;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2416640L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr4[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr4[static_cast<long>(x0)] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, ), (1, ))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_53, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, ), (1, ))
    assert_size_stride(primals_65, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_74, (2, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (), ())
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_80, (), ())
    assert_size_stride(primals_81, (128, ), (1, ))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (), ())
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (), ())
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (), ())
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (), ())
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_95, (), ())
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (), ())
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_101, (), ())
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_104, (), ())
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_107, (), ())
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, ), (1, ))
    assert_size_stride(primals_110, (), ())
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (), ())
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (), ())
    assert_size_stride(primals_117, (128, ), (1, ))
    assert_size_stride(primals_118, (128, ), (1, ))
    assert_size_stride(primals_119, (), ())
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (), ())
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (64, ), (1, ))
    assert_size_stride(primals_125, (), ())
    assert_size_stride(primals_126, (64, ), (1, ))
    assert_size_stride(primals_127, (64, ), (1, ))
    assert_size_stride(primals_128, (), ())
    assert_size_stride(primals_129, (2, 3, 640, 959), (1841280, 613760, 959, 1))
    buf0 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((512, 1024, 3, 3), (9216, 1, 3072, 1024), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((256, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((256, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((64, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((64, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((2, 3, 640, 959), (1841280, 1, 2877, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del primals_1
    del primals_129
    del primals_13
    del primals_17
    del primals_21
    del primals_25
    del primals_29
    del primals_33
    del primals_37
    del primals_41
    del primals_45
    del primals_49
    del primals_5
    del primals_53
    del primals_57
    del primals_61
    del primals_65
    del primals_69
    del primals_9
    # Source Nodes: [l__mod___inc_double_conv_0], Original ATen: [aten.convolution]
    buf19 = extern_kernels.convolution(buf18, buf0, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf19, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    del primals_2
    buf20 = empty_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf19.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf20.data_ptr()))
    del primals_4
    # Source Nodes: [l__mod___inc_double_conv_3], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(buf20, buf1, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf21, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    del primals_6
    buf22 = empty_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((2, 64, 320, 479), (9809920, 1, 30656, 64), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((2, 64, 320, 479), (9809920, 1, 30656, 64), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_2(c_void_p(buf21.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()))
    del primals_8
    # Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_0], Original ATen: [aten.convolution]
    buf25 = extern_kernels.convolution(buf23, buf2, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf25, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    del primals_10
    buf26 = empty_strided((2, 128, 320, 479), (19619840, 1, 61312, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_3(c_void_p(buf25.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf26.data_ptr()))
    del primals_12
    # Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_3], Original ATen: [aten.convolution]
    buf27 = extern_kernels.convolution(buf26, buf3, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf27, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    del primals_14
    buf28 = empty_strided((2, 128, 320, 479), (19619840, 1, 61312, 128), device='cpu', dtype=torch.float32)
    buf29 = empty_strided((2, 128, 160, 239), (4894720, 1, 30592, 128), device='cpu', dtype=torch.float32)
    buf30 = empty_strided((2, 128, 160, 239), (4894720, 1, 30592, 128), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_4(c_void_p(buf27.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    del primals_16
    # Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_0], Original ATen: [aten.convolution]
    buf31 = extern_kernels.convolution(buf29, buf4, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf31, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    del primals_18
    buf32 = empty_strided((2, 256, 160, 239), (9789440, 1, 61184, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_5(c_void_p(buf31.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf32.data_ptr()))
    del primals_20
    # Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_3], Original ATen: [aten.convolution]
    buf33 = extern_kernels.convolution(buf32, buf5, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf33, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    del primals_22
    buf34 = empty_strided((2, 256, 160, 239), (9789440, 1, 61184, 256), device='cpu', dtype=torch.float32)
    buf35 = empty_strided((2, 256, 80, 119), (2437120, 1, 30464, 256), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((2, 256, 80, 119), (2437120, 1, 30464, 256), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_6(c_void_p(buf33.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del primals_24
    # Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_0], Original ATen: [aten.convolution]
    buf37 = extern_kernels.convolution(buf35, buf6, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf37, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    del primals_26
    buf38 = empty_strided((2, 512, 80, 119), (4874240, 1, 60928, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_7(c_void_p(buf37.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf38.data_ptr()))
    del primals_28
    # Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_3], Original ATen: [aten.convolution]
    buf39 = extern_kernels.convolution(buf38, buf7, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf39, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    del primals_30
    buf40 = empty_strided((2, 512, 80, 119), (4874240, 1, 60928, 512), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.float32)
    buf42 = empty_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_8(c_void_p(buf39.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()))
    del primals_32
    # Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_0], Original ATen: [aten.convolution]
    buf43 = extern_kernels.convolution(buf41, buf8, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf43, (2, 512, 40, 59), (1208320, 1, 30208, 512))
    del primals_34
    buf44 = empty_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_9(c_void_p(buf43.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf44.data_ptr()))
    del primals_36
    # Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_3], Original ATen: [aten.convolution]
    buf45 = extern_kernels.convolution(buf44, buf9, primals_38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf45, (2, 512, 40, 59), (1208320, 1, 30208, 512))
    del primals_38
    buf46 = empty_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.float32)
    buf47 = empty((118, ), device='cpu', dtype=torch.int64)
    buf48 = empty((118, ), device='cpu', dtype=torch.int64)
    buf49 = empty((80, 1), device='cpu', dtype=torch.int64)
    buf50 = empty((80, 1), device='cpu', dtype=torch.int64)
    buf51 = empty((80, 1), device='cpu', dtype=torch.float32)
    buf52 = empty((118, ), device='cpu', dtype=torch.float32)
    buf53 = empty((2, 512, 80, 118), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((2, 1024, 80, 119), (9748480, 1, 121856, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_cat_ceil_clamp_mul_relu_rsub_sub_unsqueeze_10(c_void_p(buf45.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del buf53
    del primals_40
    # Source Nodes: [l__mod___up1_conv_double_conv_0], Original ATen: [aten.convolution]
    buf55 = extern_kernels.convolution(buf54, buf10, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf55, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    del primals_42
    buf56 = empty_strided((2, 512, 80, 119), (4874240, 1, 60928, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_11(c_void_p(buf55.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf56.data_ptr()))
    del primals_44
    # Source Nodes: [l__mod___up1_conv_double_conv_3], Original ATen: [aten.convolution]
    buf57 = extern_kernels.convolution(buf56, buf11, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf57, (2, 256, 80, 119), (2437120, 1, 30464, 256))
    del primals_46
    buf58 = empty_strided((2, 256, 80, 119), (2437120, 1, 30464, 256), device='cpu', dtype=torch.float32)
    buf59 = empty((238, ), device='cpu', dtype=torch.int64)
    buf60 = empty((238, ), device='cpu', dtype=torch.int64)
    buf61 = empty((160, 1), device='cpu', dtype=torch.int64)
    buf62 = empty((160, 1), device='cpu', dtype=torch.int64)
    buf63 = empty((160, 1), device='cpu', dtype=torch.float32)
    buf64 = empty((238, ), device='cpu', dtype=torch.float32)
    buf65 = empty((2, 256, 160, 238), device='cpu', dtype=torch.float32)
    buf66 = empty_strided((2, 512, 160, 239), (19578880, 1, 122368, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_cat_ceil_clamp_mul_relu_rsub_sub_unsqueeze_12(c_void_p(buf57.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    del buf65
    del primals_48
    # Source Nodes: [l__mod___up2_conv_double_conv_0], Original ATen: [aten.convolution]
    buf67 = extern_kernels.convolution(buf66, buf12, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf67, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    del primals_50
    buf68 = empty_strided((2, 256, 160, 239), (9789440, 1, 61184, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_13(c_void_p(buf67.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf68.data_ptr()))
    del primals_52
    # Source Nodes: [l__mod___up2_conv_double_conv_3], Original ATen: [aten.convolution]
    buf69 = extern_kernels.convolution(buf68, buf13, primals_54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf69, (2, 128, 160, 239), (4894720, 1, 30592, 128))
    del primals_54
    buf70 = empty_strided((2, 128, 160, 239), (4894720, 1, 30592, 128), device='cpu', dtype=torch.float32)
    buf71 = empty((478, ), device='cpu', dtype=torch.int64)
    buf72 = empty((478, ), device='cpu', dtype=torch.int64)
    buf73 = empty((320, 1), device='cpu', dtype=torch.int64)
    buf74 = empty((320, 1), device='cpu', dtype=torch.int64)
    buf75 = empty((320, 1), device='cpu', dtype=torch.float32)
    buf76 = empty((478, ), device='cpu', dtype=torch.float32)
    buf77 = empty((2, 128, 320, 478), device='cpu', dtype=torch.float32)
    buf78 = empty_strided((2, 256, 320, 479), (39239680, 1, 122624, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_cat_ceil_clamp_mul_relu_rsub_sub_unsqueeze_14(c_void_p(buf69.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del buf77
    del primals_56
    # Source Nodes: [l__mod___up3_conv_double_conv_0], Original ATen: [aten.convolution]
    buf79 = extern_kernels.convolution(buf78, buf14, primals_58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf79, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    del primals_58
    buf80 = empty_strided((2, 128, 320, 479), (19619840, 1, 61312, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_15(c_void_p(buf79.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf80.data_ptr()))
    del primals_60
    # Source Nodes: [l__mod___up3_conv_double_conv_3], Original ATen: [aten.convolution]
    buf81 = extern_kernels.convolution(buf80, buf15, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf81, (2, 64, 320, 479), (9809920, 1, 30656, 64))
    del primals_62
    buf82 = empty_strided((2, 64, 320, 479), (9809920, 1, 30656, 64), device='cpu', dtype=torch.float32)
    buf83 = empty((958, ), device='cpu', dtype=torch.int64)
    buf84 = empty((958, ), device='cpu', dtype=torch.int64)
    buf85 = empty((640, 1), device='cpu', dtype=torch.int64)
    buf86 = empty((640, 1), device='cpu', dtype=torch.int64)
    buf87 = empty((640, 1), device='cpu', dtype=torch.float32)
    buf88 = empty((958, ), device='cpu', dtype=torch.float32)
    buf89 = empty((2, 64, 640, 958), device='cpu', dtype=torch.float32)
    buf90 = empty_strided((2, 128, 640, 959), (78561280, 1, 122752, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_cat_ceil_clamp_mul_relu_rsub_sub_unsqueeze_16(c_void_p(buf81.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del buf89
    del primals_64
    # Source Nodes: [l__mod___up4_conv_double_conv_0], Original ATen: [aten.convolution]
    buf91 = extern_kernels.convolution(buf90, buf16, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf91, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    del primals_66
    buf92 = empty_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_17(c_void_p(buf91.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf92.data_ptr()))
    del primals_68
    # Source Nodes: [l__mod___up4_conv_double_conv_3], Original ATen: [aten.convolution]
    buf93 = extern_kernels.convolution(buf92, buf17, primals_70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf93, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    del primals_70
    buf94 = empty_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_18(c_void_p(buf93.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf94.data_ptr()))
    del primals_72
    # Source Nodes: [pred], Original ATen: [aten.convolution]
    buf95 = extern_kernels.convolution(buf94, primals_73, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf95, (2, 2, 640, 959), (1227520, 1, 1918, 2))
    del primals_74
    buf96 = empty((2, 2, 640, 959), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((2, 64, 320, 479), (9809920, 1, 30656, 64), device='cpu', dtype=torch.bool)
    buf98 = empty_strided((2, 128, 160, 239), (4894720, 1, 30592, 128), device='cpu', dtype=torch.bool)
    buf99 = empty_strided((2, 256, 80, 119), (2437120, 1, 30464, 256), device='cpu', dtype=torch.bool)
    buf100 = empty_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.bool)
    cpp_fused_convolution_threshold_backward_19(c_void_p(buf95.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()))
    return (buf96, buf0, primals_3, buf1, primals_7, buf2, primals_11, buf3, primals_15, buf4, primals_19, buf5, primals_23, buf6, primals_27, buf7, primals_31, buf8, primals_35, buf9, primals_39, buf10, primals_43, buf11, primals_47, buf12, primals_51, buf13, primals_55, buf14, primals_59, buf15, primals_63, buf16, primals_67, buf17, primals_71, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf47, buf48, buf49, buf50, buf51, buf52, buf54, buf55, buf56, buf57, buf59, buf60, buf61, buf62, buf63, buf64, buf66, buf67, buf68, buf69, buf71, buf72, buf73, buf74, buf75, buf76, buf78, buf79, buf80, buf81, buf83, buf84, buf85, buf86, buf87, buf88, buf90, buf91, buf92, buf93, buf94, buf97, buf98, buf99, buf100, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_78 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_81 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_84 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_87 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_90 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_93 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_96 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_99 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_102 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_105 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_108 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_111 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_114 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_117 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_120 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_123 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_126 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_129 = rand_strided((2, 3, 640, 959), (1841280, 613760, 959, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pytorch_unet', benchmark_compiled_module)
