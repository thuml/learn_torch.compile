
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
                       float* out_ptr13)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr8[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(65536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr13[static_cast<long>(x2 + (65536L*x1) + (196608L*x0))];
                        out_ptr13[static_cast<long>(x1 + (3L*x2) + (196608L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(131072.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(131072.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(32768.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32768.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(32768.0);
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
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_4 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(32768.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(32768.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp18 = tmp16 - tmp17;
                    auto tmp20 = tmp19 / tmp5;
                    auto tmp21 = tmp20 + tmp8;
                    auto tmp22 = tmp21.rsqrt();
                    auto tmp23 = tmp18 * tmp22;
                    auto tmp25 = tmp23 * tmp24;
                    auto tmp27 = tmp25 + tmp26;
                    auto tmp28 = tmp15 + tmp27;
                    tmp28.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(8192.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
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
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_7 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(8192.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp18 = tmp16 - tmp17;
                    auto tmp20 = tmp19 / tmp5;
                    auto tmp21 = tmp20 + tmp8;
                    auto tmp22 = tmp21.rsqrt();
                    auto tmp23 = tmp18 * tmp22;
                    auto tmp25 = tmp23 * tmp24;
                    auto tmp27 = tmp25 + tmp26;
                    auto tmp28 = tmp15 + tmp27;
                    tmp28.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(8192.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(8192.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(8192.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_13 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp18 = tmp16 - tmp17;
                    auto tmp20 = tmp19 / tmp5;
                    auto tmp21 = tmp20 + tmp8;
                    auto tmp22 = tmp21.rsqrt();
                    auto tmp23 = tmp18 * tmp22;
                    auto tmp25 = tmp23 * tmp24;
                    auto tmp27 = tmp25 + tmp26;
                    auto tmp28 = tmp15 + tmp27;
                    tmp28.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_32 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp18 = tmp16 - tmp17;
                    auto tmp20 = tmp19 / tmp5;
                    auto tmp21 = tmp20 + tmp8;
                    auto tmp22 = tmp21.rsqrt();
                    auto tmp23 = tmp18 * tmp22;
                    auto tmp25 = tmp23 * tmp24;
                    auto tmp27 = tmp25 + tmp26;
                    auto tmp28 = tmp15 + tmp27;
                    tmp28.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = at::vec::clamp_min(tmp17, decltype(tmp17)(0));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2560L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2560L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2560L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(512.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (2560L*x2) + (163840L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (2560L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(20480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_threshold_backward_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       float* in_out_ptr4,
                       float* in_out_ptr5,
                       float* in_out_ptr6,
                       float* in_out_ptr7,
                       float* in_out_ptr8,
                       float* in_out_ptr9,
                       float* in_out_ptr10,
                       float* in_out_ptr11,
                       float* in_out_ptr12,
                       float* in_out_ptr13,
                       float* in_out_ptr14,
                       float* in_out_ptr15,
                       float* in_out_ptr16,
                       float* in_out_ptr17,
                       float* in_out_ptr18,
                       float* in_out_ptr19,
                       float* in_out_ptr20,
                       float* in_out_ptr21,
                       float* in_out_ptr22,
                       float* in_out_ptr23,
                       float* in_out_ptr24,
                       float* in_out_ptr25,
                       float* in_out_ptr26,
                       float* in_out_ptr27,
                       float* in_out_ptr28,
                       float* in_out_ptr29,
                       float* in_out_ptr30,
                       float* in_out_ptr31,
                       float* in_out_ptr32,
                       float* in_out_ptr33,
                       float* in_out_ptr34,
                       float* in_out_ptr35,
                       float* in_out_ptr36,
                       float* in_out_ptr37,
                       float* in_out_ptr38,
                       float* in_out_ptr39,
                       float* in_out_ptr40,
                       float* in_out_ptr41,
                       float* in_out_ptr42,
                       float* in_out_ptr43,
                       float* in_out_ptr44,
                       float* in_out_ptr45,
                       float* in_out_ptr46,
                       float* in_out_ptr47,
                       float* in_out_ptr48,
                       float* in_out_ptr49,
                       float* in_out_ptr50,
                       float* in_out_ptr51,
                       float* in_out_ptr52,
                       float* in_out_ptr53,
                       float* in_out_ptr54,
                       float* in_out_ptr55,
                       float* in_out_ptr56,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const long* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const long* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const long* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const long* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const long* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const long* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
                       const long* in_ptr33,
                       const float* in_ptr34,
                       const float* in_ptr35,
                       const float* in_ptr36,
                       const long* in_ptr37,
                       const float* in_ptr38,
                       const float* in_ptr39,
                       const float* in_ptr40,
                       const long* in_ptr41,
                       const float* in_ptr42,
                       const float* in_ptr43,
                       const float* in_ptr44,
                       const long* in_ptr45,
                       const float* in_ptr46,
                       const float* in_ptr47,
                       const float* in_ptr48,
                       const long* in_ptr49,
                       const float* in_ptr50,
                       const float* in_ptr51,
                       const float* in_ptr52,
                       const long* in_ptr53,
                       const float* in_ptr54,
                       const float* in_ptr55,
                       const float* in_ptr56,
                       const long* in_ptr57,
                       const float* in_ptr58,
                       const float* in_ptr59,
                       const float* in_ptr60,
                       const long* in_ptr61,
                       const float* in_ptr62,
                       const float* in_ptr63,
                       const float* in_ptr64,
                       const long* in_ptr65,
                       const float* in_ptr66,
                       const float* in_ptr67,
                       const float* in_ptr68,
                       const long* in_ptr69,
                       const float* in_ptr70,
                       const float* in_ptr71,
                       const float* in_ptr72,
                       const long* in_ptr73,
                       const float* in_ptr74,
                       const float* in_ptr75,
                       const float* in_ptr76,
                       const long* in_ptr77,
                       const float* in_ptr78,
                       const float* in_ptr79,
                       const float* in_ptr80,
                       const long* in_ptr81,
                       const float* in_ptr82,
                       const float* in_ptr83,
                       const float* in_ptr84,
                       const long* in_ptr85,
                       const float* in_ptr86,
                       const float* in_ptr87,
                       const float* in_ptr88,
                       const long* in_ptr89,
                       const float* in_ptr90,
                       const float* in_ptr91,
                       const float* in_ptr92,
                       const long* in_ptr93,
                       const float* in_ptr94,
                       const float* in_ptr95,
                       const float* in_ptr96,
                       const long* in_ptr97,
                       const float* in_ptr98,
                       const float* in_ptr99,
                       const float* in_ptr100,
                       const long* in_ptr101,
                       const float* in_ptr102,
                       const float* in_ptr103,
                       const float* in_ptr104,
                       const long* in_ptr105,
                       const float* in_ptr106,
                       const float* in_ptr107,
                       const float* in_ptr108,
                       const long* in_ptr109,
                       const float* in_ptr110,
                       const float* in_ptr111,
                       const float* in_ptr112,
                       const long* in_ptr113,
                       const float* in_ptr114,
                       const float* in_ptr115,
                       const float* in_ptr116,
                       const long* in_ptr117,
                       const float* in_ptr118,
                       const float* in_ptr119,
                       const float* in_ptr120,
                       const long* in_ptr121,
                       const float* in_ptr122,
                       const float* in_ptr123,
                       const float* in_ptr124,
                       const long* in_ptr125,
                       const float* in_ptr126,
                       const float* in_ptr127,
                       const float* in_ptr128,
                       const long* in_ptr129,
                       const float* in_ptr130,
                       const float* in_ptr131,
                       const float* in_ptr132,
                       const long* in_ptr133,
                       const float* in_ptr134,
                       const float* in_ptr135,
                       const float* in_ptr136,
                       const long* in_ptr137,
                       const float* in_ptr138,
                       const float* in_ptr139,
                       const float* in_ptr140,
                       const long* in_ptr141,
                       const float* in_ptr142,
                       const float* in_ptr143,
                       const float* in_ptr144,
                       const long* in_ptr145,
                       const float* in_ptr146,
                       const float* in_ptr147,
                       const float* in_ptr148,
                       const long* in_ptr149,
                       const float* in_ptr150,
                       const float* in_ptr151,
                       const float* in_ptr152,
                       const long* in_ptr153,
                       const float* in_ptr154,
                       const float* in_ptr155,
                       const float* in_ptr156,
                       const long* in_ptr157,
                       const float* in_ptr158,
                       const float* in_ptr159,
                       const float* in_ptr160,
                       const long* in_ptr161,
                       const float* in_ptr162,
                       const float* in_ptr163,
                       const float* in_ptr164,
                       const long* in_ptr165,
                       const float* in_ptr166,
                       const float* in_ptr167,
                       const float* in_ptr168,
                       const long* in_ptr169,
                       const float* in_ptr170,
                       const float* in_ptr171,
                       const float* in_ptr172,
                       const long* in_ptr173,
                       const float* in_ptr174,
                       const float* in_ptr175,
                       const float* in_ptr176,
                       const long* in_ptr177,
                       const float* in_ptr178,
                       const float* in_ptr179,
                       const float* in_ptr180,
                       const long* in_ptr181,
                       const float* in_ptr182,
                       const float* in_ptr183,
                       const float* in_ptr184,
                       const long* in_ptr185,
                       const float* in_ptr186,
                       const float* in_ptr187,
                       const float* in_ptr188,
                       const long* in_ptr189,
                       const float* in_ptr190,
                       const float* in_ptr191,
                       const float* in_ptr192,
                       const long* in_ptr193,
                       const float* in_ptr194,
                       const float* in_ptr195,
                       const float* in_ptr196,
                       const long* in_ptr197,
                       const float* in_ptr198,
                       const float* in_ptr199,
                       const float* in_ptr200,
                       const long* in_ptr201,
                       const float* in_ptr202,
                       const float* in_ptr203,
                       const float* in_ptr204,
                       const long* in_ptr205,
                       const float* in_ptr206,
                       const float* in_ptr207,
                       const float* in_ptr208,
                       const long* in_ptr209,
                       const float* in_ptr210,
                       const float* in_ptr211,
                       const float* in_ptr212,
                       const long* in_ptr213,
                       const float* in_ptr214,
                       const float* in_ptr215,
                       const float* in_ptr216,
                       const long* in_ptr217,
                       const float* in_ptr218,
                       const float* in_ptr219,
                       const float* in_ptr220,
                       const long* in_ptr221,
                       const float* in_ptr222,
                       const float* in_ptr223,
                       const float* in_ptr224,
                       const long* in_ptr225,
                       const float* in_ptr226,
                       const float* in_ptr227,
                       const float* in_ptr228,
                       bool* out_ptr0,
                       long* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5,
                       long* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       long* out_ptr12,
                       float* out_ptr14,
                       float* out_ptr15,
                       long* out_ptr17,
                       float* out_ptr19,
                       float* out_ptr20,
                       long* out_ptr22,
                       float* out_ptr24,
                       float* out_ptr25,
                       long* out_ptr27,
                       float* out_ptr29,
                       float* out_ptr30,
                       long* out_ptr32,
                       float* out_ptr34,
                       float* out_ptr35,
                       long* out_ptr37,
                       float* out_ptr39,
                       float* out_ptr40,
                       long* out_ptr42,
                       float* out_ptr44,
                       float* out_ptr45,
                       long* out_ptr47,
                       float* out_ptr49,
                       float* out_ptr50,
                       long* out_ptr52,
                       float* out_ptr54,
                       float* out_ptr55,
                       long* out_ptr57,
                       float* out_ptr59,
                       float* out_ptr60,
                       long* out_ptr62,
                       float* out_ptr64,
                       float* out_ptr65,
                       long* out_ptr67,
                       float* out_ptr69,
                       float* out_ptr70,
                       long* out_ptr72,
                       float* out_ptr74,
                       float* out_ptr75,
                       long* out_ptr77,
                       float* out_ptr79,
                       float* out_ptr80,
                       long* out_ptr82,
                       float* out_ptr84,
                       float* out_ptr85,
                       long* out_ptr87,
                       float* out_ptr89,
                       float* out_ptr90,
                       long* out_ptr92,
                       float* out_ptr94,
                       float* out_ptr95,
                       long* out_ptr97,
                       float* out_ptr99,
                       float* out_ptr100,
                       long* out_ptr102,
                       float* out_ptr104,
                       float* out_ptr105,
                       long* out_ptr107,
                       float* out_ptr109,
                       float* out_ptr110,
                       long* out_ptr112,
                       float* out_ptr114,
                       float* out_ptr115,
                       long* out_ptr117,
                       float* out_ptr119,
                       float* out_ptr120,
                       long* out_ptr122,
                       float* out_ptr124,
                       float* out_ptr125,
                       long* out_ptr127,
                       float* out_ptr129,
                       float* out_ptr130,
                       long* out_ptr132,
                       float* out_ptr134,
                       float* out_ptr135,
                       long* out_ptr137,
                       float* out_ptr139,
                       float* out_ptr140,
                       long* out_ptr142,
                       float* out_ptr144,
                       float* out_ptr145,
                       long* out_ptr147,
                       float* out_ptr149,
                       float* out_ptr150,
                       long* out_ptr152,
                       float* out_ptr154,
                       float* out_ptr155,
                       long* out_ptr157,
                       float* out_ptr159,
                       float* out_ptr160,
                       long* out_ptr162,
                       float* out_ptr164,
                       float* out_ptr165,
                       long* out_ptr167,
                       float* out_ptr169,
                       float* out_ptr170,
                       long* out_ptr172,
                       float* out_ptr174,
                       float* out_ptr175,
                       long* out_ptr177,
                       float* out_ptr179,
                       float* out_ptr180,
                       long* out_ptr182,
                       float* out_ptr184,
                       float* out_ptr185,
                       long* out_ptr187,
                       float* out_ptr189,
                       float* out_ptr190,
                       long* out_ptr192,
                       float* out_ptr194,
                       float* out_ptr195,
                       long* out_ptr197,
                       float* out_ptr199,
                       float* out_ptr200,
                       long* out_ptr202,
                       float* out_ptr204,
                       float* out_ptr205,
                       long* out_ptr207,
                       float* out_ptr209,
                       float* out_ptr210,
                       long* out_ptr212,
                       float* out_ptr214,
                       float* out_ptr215,
                       long* out_ptr217,
                       float* out_ptr219,
                       float* out_ptr220,
                       long* out_ptr222,
                       float* out_ptr224,
                       float* out_ptr225,
                       long* out_ptr227,
                       float* out_ptr229,
                       float* out_ptr230,
                       long* out_ptr232,
                       float* out_ptr234,
                       float* out_ptr235,
                       long* out_ptr237,
                       float* out_ptr239,
                       float* out_ptr240,
                       long* out_ptr242,
                       float* out_ptr244,
                       float* out_ptr245,
                       long* out_ptr247,
                       float* out_ptr249,
                       float* out_ptr250,
                       long* out_ptr252,
                       float* out_ptr254,
                       float* out_ptr255,
                       long* out_ptr257,
                       float* out_ptr259,
                       float* out_ptr260,
                       long* out_ptr262,
                       float* out_ptr264,
                       float* out_ptr265,
                       long* out_ptr267,
                       float* out_ptr269,
                       float* out_ptr270,
                       long* out_ptr272,
                       float* out_ptr274,
                       float* out_ptr275,
                       long* out_ptr277,
                       float* out_ptr279,
                       float* out_ptr280,
                       long* out_ptr282,
                       float* out_ptr284,
                       float* out_ptr285)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr1[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr2[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(131072.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000076294527394);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr5[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr7[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr9 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(32768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.000030518509476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr10 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr9[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr12[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr14 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(32768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.000030518509476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr13[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr17[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr19 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(32768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.000030518509476);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr20 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr17[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr22[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr24 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr24 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001220852154804);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr25 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr21[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr27[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr29 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr29 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001220852154804);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr30 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr25[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr32[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr34 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr34 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr35 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001220852154804);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr35 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr29[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr37[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr30 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr39 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr39 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr40 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001220852154804);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr40 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr33[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr42[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr34 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr44 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr44 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr45 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001220852154804);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr45 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr37[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr47[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr38 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr49 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr49 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr50 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001220852154804);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr50 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr41[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr52[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr42 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr54 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr54 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr55 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr55 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr45[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr57[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr46 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr59 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr59 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr60 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr60 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr49[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr62[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr50 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr64 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr64 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr65 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr65 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr53[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr67[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr54 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr69 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr69 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr70 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr70 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr57[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr72[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr58 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr74 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr74 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr75 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr75 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr61[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr77[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr62 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr79 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr79 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr80 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr80 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr65[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr82[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr66 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr84 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr84 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr85 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr85 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr69[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr87[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr70 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr89 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr89 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr90 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr90 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr73[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr92[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr74 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr94 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr94 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr95 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr95 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr77[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr97[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr78 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr99 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr99 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr100 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr100 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr81[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr102[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr82 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr104 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr104 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr105 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr105 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr85[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr107[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr86 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr109 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr109 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr110 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr110 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr89[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr112[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr90 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr114 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr114 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr115 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr115 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr93[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr117[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr94 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr119 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr119 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr120 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr120 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr97[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr122[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr98 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr124 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr124 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr125 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr125 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr101[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr127[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr102 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr129 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr129 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr130 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr130 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr105[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr132[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr106 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr134 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr134 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr135 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr135 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr109[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr137[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr110 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr139 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr139 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr140 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr140 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr113[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr142[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr114 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr144 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr144 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr145 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0004885197850513);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr145 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr117[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr147[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr118 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr149 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr149 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr150 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr150 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr121[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr152[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr122 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr154 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr154 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr155 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr155 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr125[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr157[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr126 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr159 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr159 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr160 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr160 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr129[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr162[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr130 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr164 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr164 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr165 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr165 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr133[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr167[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr134 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr169 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr169 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr170 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr170 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr137[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr172[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr138 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr174 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr174 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr175 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr175 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr141[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr177[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr142 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr179 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr179 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr180 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr180 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr145[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr182[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr146 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr184 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr184 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr185 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr185 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr149[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr187[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr150 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr189 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr189 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr190 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr190 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr153[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr192[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr154 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr194 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr194 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr195 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr195 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr157[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr197[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr158 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr199 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr199 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr200 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr200 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr161[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr202[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr162 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr204 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr204 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr205 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr205 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr165[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr207[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr166 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr209 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr209 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr210 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr210 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr169[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr212[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr170 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr214 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr214 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr215 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr215 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr173[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr217[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr174 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr219 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr219 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr220 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr220 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr177[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr222[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr178 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr224 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr224 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr225 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr225 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr181[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr227[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr182 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr229 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr229 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr230 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr230 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr185[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr232[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr186 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr234 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr234 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr235 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr235 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr189[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr237[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr190 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr239 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr239 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr240 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr240 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr193[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr242[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr194 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr244 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr244 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr245 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr245 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr197[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr247[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr198 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr249 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr249 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr250 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr250 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr201[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr252[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr202 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr254 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr254 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr255 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr255 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr205[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr257[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr206 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr259 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr259 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr260 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr260 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr209[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr262[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr210 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr264 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr264 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr265 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr265 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr213[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr267[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr214 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr269 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr269 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr270 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr270 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr217[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr272[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr218 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr274 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr274 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr275 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr275 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr221[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr277[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr222 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr279 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr279 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr55 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr280 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr280 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr225[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr282[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2560L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr226 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr284 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr284 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2560L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr285 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0019569471624266);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr285 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_10, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_12, (192, ), (1, ))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_16, (192, ), (1, ))
    assert_size_stride(primals_17, (192, ), (1, ))
    assert_size_stride(primals_18, (192, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_20, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_22, (160, ), (1, ))
    assert_size_stride(primals_23, (640, ), (1, ))
    assert_size_stride(primals_24, (640, ), (1, ))
    assert_size_stride(primals_25, (640, ), (1, ))
    assert_size_stride(primals_26, (640, ), (1, ))
    assert_size_stride(primals_27, (160, ), (1, ))
    assert_size_stride(primals_28, (160, ), (1, ))
    assert_size_stride(primals_29, (160, ), (1, ))
    assert_size_stride(primals_30, (160, ), (1, ))
    assert_size_stride(primals_31, (640, ), (1, ))
    assert_size_stride(primals_32, (640, ), (1, ))
    assert_size_stride(primals_33, (160, ), (1, ))
    assert_size_stride(primals_34, (160, ), (1, ))
    assert_size_stride(primals_35, (160, ), (1, ))
    assert_size_stride(primals_36, (160, ), (1, ))
    assert_size_stride(primals_37, (640, ), (1, ))
    assert_size_stride(primals_38, (640, ), (1, ))
    assert_size_stride(primals_39, (160, ), (1, ))
    assert_size_stride(primals_40, (160, ), (1, ))
    assert_size_stride(primals_41, (160, ), (1, ))
    assert_size_stride(primals_42, (160, ), (1, ))
    assert_size_stride(primals_43, (640, ), (1, ))
    assert_size_stride(primals_44, (640, ), (1, ))
    assert_size_stride(primals_45, (160, ), (1, ))
    assert_size_stride(primals_46, (160, ), (1, ))
    assert_size_stride(primals_47, (160, ), (1, ))
    assert_size_stride(primals_48, (160, ), (1, ))
    assert_size_stride(primals_49, (640, ), (1, ))
    assert_size_stride(primals_50, (640, ), (1, ))
    assert_size_stride(primals_51, (160, ), (1, ))
    assert_size_stride(primals_52, (160, ), (1, ))
    assert_size_stride(primals_53, (160, ), (1, ))
    assert_size_stride(primals_54, (160, ), (1, ))
    assert_size_stride(primals_55, (640, ), (1, ))
    assert_size_stride(primals_56, (640, ), (1, ))
    assert_size_stride(primals_57, (1920, ), (1, ))
    assert_size_stride(primals_58, (1920, ), (1, ))
    assert_size_stride(primals_59, (1920, ), (1, ))
    assert_size_stride(primals_60, (1920, ), (1, ))
    assert_size_stride(primals_61, (640, ), (1, ))
    assert_size_stride(primals_62, (640, ), (1, ))
    assert_size_stride(primals_63, (640, ), (1, ))
    assert_size_stride(primals_64, (640, ), (1, ))
    assert_size_stride(primals_65, (1920, ), (1, ))
    assert_size_stride(primals_66, (1920, ), (1, ))
    assert_size_stride(primals_67, (1920, ), (1, ))
    assert_size_stride(primals_68, (1920, ), (1, ))
    assert_size_stride(primals_69, (640, ), (1, ))
    assert_size_stride(primals_70, (640, ), (1, ))
    assert_size_stride(primals_71, (1920, ), (1, ))
    assert_size_stride(primals_72, (1920, ), (1, ))
    assert_size_stride(primals_73, (1920, ), (1, ))
    assert_size_stride(primals_74, (1920, ), (1, ))
    assert_size_stride(primals_75, (640, ), (1, ))
    assert_size_stride(primals_76, (640, ), (1, ))
    assert_size_stride(primals_77, (1920, ), (1, ))
    assert_size_stride(primals_78, (1920, ), (1, ))
    assert_size_stride(primals_79, (1920, ), (1, ))
    assert_size_stride(primals_80, (1920, ), (1, ))
    assert_size_stride(primals_81, (640, ), (1, ))
    assert_size_stride(primals_82, (640, ), (1, ))
    assert_size_stride(primals_83, (1920, ), (1, ))
    assert_size_stride(primals_84, (1920, ), (1, ))
    assert_size_stride(primals_85, (1920, ), (1, ))
    assert_size_stride(primals_86, (1920, ), (1, ))
    assert_size_stride(primals_87, (640, ), (1, ))
    assert_size_stride(primals_88, (640, ), (1, ))
    assert_size_stride(primals_89, (1920, ), (1, ))
    assert_size_stride(primals_90, (1920, ), (1, ))
    assert_size_stride(primals_91, (1920, ), (1, ))
    assert_size_stride(primals_92, (1920, ), (1, ))
    assert_size_stride(primals_93, (640, ), (1, ))
    assert_size_stride(primals_94, (640, ), (1, ))
    assert_size_stride(primals_95, (1920, ), (1, ))
    assert_size_stride(primals_96, (1920, ), (1, ))
    assert_size_stride(primals_97, (1920, ), (1, ))
    assert_size_stride(primals_98, (1920, ), (1, ))
    assert_size_stride(primals_99, (640, ), (1, ))
    assert_size_stride(primals_100, (640, ), (1, ))
    assert_size_stride(primals_101, (1920, ), (1, ))
    assert_size_stride(primals_102, (1920, ), (1, ))
    assert_size_stride(primals_103, (1920, ), (1, ))
    assert_size_stride(primals_104, (1920, ), (1, ))
    assert_size_stride(primals_105, (640, ), (1, ))
    assert_size_stride(primals_106, (640, ), (1, ))
    assert_size_stride(primals_107, (1920, ), (1, ))
    assert_size_stride(primals_108, (1920, ), (1, ))
    assert_size_stride(primals_109, (1920, ), (1, ))
    assert_size_stride(primals_110, (1920, ), (1, ))
    assert_size_stride(primals_111, (640, ), (1, ))
    assert_size_stride(primals_112, (640, ), (1, ))
    assert_size_stride(primals_113, (2560, ), (1, ))
    assert_size_stride(primals_114, (2560, ), (1, ))
    assert_size_stride(primals_115, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_116, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_117, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_118, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_119, (192, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_120, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_121, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_122, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_123, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_124, (160, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_125, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_126, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_127, (640, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_128, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_129, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_130, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_131, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_132, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_133, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_134, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_135, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_136, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_137, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_138, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_139, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_140, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_141, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_142, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_143, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_144, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_145, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_146, (640, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_147, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_148, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_150, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_151, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_153, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_154, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_156, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_157, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_158, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_159, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_160, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_162, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_163, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_164, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_165, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_166, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_167, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_168, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_169, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_170, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_171, (2560, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_172, (1000, 2560), (2560, 1))
    assert_size_stride(primals_173, (1000, ), (1, ))
    assert_size_stride(primals_174, (), ())
    assert_size_stride(primals_175, (32, ), (1, ))
    assert_size_stride(primals_176, (32, ), (1, ))
    assert_size_stride(primals_177, (), ())
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (), ())
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_183, (), ())
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_186, (), ())
    assert_size_stride(primals_187, (192, ), (1, ))
    assert_size_stride(primals_188, (192, ), (1, ))
    assert_size_stride(primals_189, (), ())
    assert_size_stride(primals_190, (192, ), (1, ))
    assert_size_stride(primals_191, (192, ), (1, ))
    assert_size_stride(primals_192, (), ())
    assert_size_stride(primals_193, (192, ), (1, ))
    assert_size_stride(primals_194, (192, ), (1, ))
    assert_size_stride(primals_195, (), ())
    assert_size_stride(primals_196, (192, ), (1, ))
    assert_size_stride(primals_197, (192, ), (1, ))
    assert_size_stride(primals_198, (), ())
    assert_size_stride(primals_199, (192, ), (1, ))
    assert_size_stride(primals_200, (192, ), (1, ))
    assert_size_stride(primals_201, (), ())
    assert_size_stride(primals_202, (160, ), (1, ))
    assert_size_stride(primals_203, (160, ), (1, ))
    assert_size_stride(primals_204, (), ())
    assert_size_stride(primals_205, (160, ), (1, ))
    assert_size_stride(primals_206, (160, ), (1, ))
    assert_size_stride(primals_207, (), ())
    assert_size_stride(primals_208, (640, ), (1, ))
    assert_size_stride(primals_209, (640, ), (1, ))
    assert_size_stride(primals_210, (), ())
    assert_size_stride(primals_211, (640, ), (1, ))
    assert_size_stride(primals_212, (640, ), (1, ))
    assert_size_stride(primals_213, (), ())
    assert_size_stride(primals_214, (160, ), (1, ))
    assert_size_stride(primals_215, (160, ), (1, ))
    assert_size_stride(primals_216, (), ())
    assert_size_stride(primals_217, (160, ), (1, ))
    assert_size_stride(primals_218, (160, ), (1, ))
    assert_size_stride(primals_219, (), ())
    assert_size_stride(primals_220, (640, ), (1, ))
    assert_size_stride(primals_221, (640, ), (1, ))
    assert_size_stride(primals_222, (), ())
    assert_size_stride(primals_223, (160, ), (1, ))
    assert_size_stride(primals_224, (160, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (160, ), (1, ))
    assert_size_stride(primals_227, (160, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (640, ), (1, ))
    assert_size_stride(primals_230, (640, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (160, ), (1, ))
    assert_size_stride(primals_233, (160, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (160, ), (1, ))
    assert_size_stride(primals_236, (160, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (640, ), (1, ))
    assert_size_stride(primals_239, (640, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (160, ), (1, ))
    assert_size_stride(primals_242, (160, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (160, ), (1, ))
    assert_size_stride(primals_245, (160, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (640, ), (1, ))
    assert_size_stride(primals_248, (640, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (160, ), (1, ))
    assert_size_stride(primals_251, (160, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (160, ), (1, ))
    assert_size_stride(primals_254, (160, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (640, ), (1, ))
    assert_size_stride(primals_257, (640, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (1920, ), (1, ))
    assert_size_stride(primals_260, (1920, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (1920, ), (1, ))
    assert_size_stride(primals_263, (1920, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (640, ), (1, ))
    assert_size_stride(primals_266, (640, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (640, ), (1, ))
    assert_size_stride(primals_269, (640, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (1920, ), (1, ))
    assert_size_stride(primals_272, (1920, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (1920, ), (1, ))
    assert_size_stride(primals_275, (1920, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (640, ), (1, ))
    assert_size_stride(primals_278, (640, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (1920, ), (1, ))
    assert_size_stride(primals_281, (1920, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (1920, ), (1, ))
    assert_size_stride(primals_284, (1920, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (640, ), (1, ))
    assert_size_stride(primals_287, (640, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (1920, ), (1, ))
    assert_size_stride(primals_290, (1920, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (1920, ), (1, ))
    assert_size_stride(primals_293, (1920, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (640, ), (1, ))
    assert_size_stride(primals_296, (640, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (1920, ), (1, ))
    assert_size_stride(primals_299, (1920, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (1920, ), (1, ))
    assert_size_stride(primals_302, (1920, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (640, ), (1, ))
    assert_size_stride(primals_305, (640, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (1920, ), (1, ))
    assert_size_stride(primals_308, (1920, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (1920, ), (1, ))
    assert_size_stride(primals_311, (1920, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (640, ), (1, ))
    assert_size_stride(primals_314, (640, ), (1, ))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (1920, ), (1, ))
    assert_size_stride(primals_317, (1920, ), (1, ))
    assert_size_stride(primals_318, (), ())
    assert_size_stride(primals_319, (1920, ), (1, ))
    assert_size_stride(primals_320, (1920, ), (1, ))
    assert_size_stride(primals_321, (), ())
    assert_size_stride(primals_322, (640, ), (1, ))
    assert_size_stride(primals_323, (640, ), (1, ))
    assert_size_stride(primals_324, (), ())
    assert_size_stride(primals_325, (1920, ), (1, ))
    assert_size_stride(primals_326, (1920, ), (1, ))
    assert_size_stride(primals_327, (), ())
    assert_size_stride(primals_328, (1920, ), (1, ))
    assert_size_stride(primals_329, (1920, ), (1, ))
    assert_size_stride(primals_330, (), ())
    assert_size_stride(primals_331, (640, ), (1, ))
    assert_size_stride(primals_332, (640, ), (1, ))
    assert_size_stride(primals_333, (), ())
    assert_size_stride(primals_334, (1920, ), (1, ))
    assert_size_stride(primals_335, (1920, ), (1, ))
    assert_size_stride(primals_336, (), ())
    assert_size_stride(primals_337, (1920, ), (1, ))
    assert_size_stride(primals_338, (1920, ), (1, ))
    assert_size_stride(primals_339, (), ())
    assert_size_stride(primals_340, (640, ), (1, ))
    assert_size_stride(primals_341, (640, ), (1, ))
    assert_size_stride(primals_342, (), ())
    assert_size_stride(primals_343, (2560, ), (1, ))
    assert_size_stride(primals_344, (2560, ), (1, ))
    assert_size_stride(primals_345, (8, 3, 256, 256), (196608, 65536, 256, 1))
    buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((192, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del primals_115
    del primals_116
    del primals_117
    del primals_119
    del primals_120
    del primals_122
    del primals_123
    del primals_125
    del primals_129
    del primals_132
    del primals_135
    del primals_138
    del primals_141
    del primals_345
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf13, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (8, 32, 128, 128), (524288, 1, 4096, 32))
    buf15 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf18 = empty((32, ), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_1(c_void_p(buf14.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    del primals_2
    # Source Nodes: [x_6], Original ATen: [aten.convolution]
    buf20 = extern_kernels.convolution(buf19, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf20, (8, 128, 64, 64), (524288, 1, 8192, 128))
    buf21 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf24 = empty((128, ), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_2(c_void_p(buf20.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del primals_4
    # Source Nodes: [x_12], Original ATen: [aten.convolution]
    buf26 = extern_kernels.convolution(buf25, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf26, (8, 128, 64, 64), (524288, 1, 8192, 128))
    buf27 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf30 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_3(c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()))
    # Source Nodes: [x_20], Original ATen: [aten.convolution]
    buf31 = extern_kernels.convolution(buf19, primals_118, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf31, (8, 128, 64, 64), (524288, 1, 8192, 128))
    buf32 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf33 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf35 = empty((128, ), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    buf37 = buf36; del buf36  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_relu_4(c_void_p(buf37.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf35.data_ptr()))
    del primals_6
    del primals_8
    # Source Nodes: [x_26], Original ATen: [aten.convolution]
    buf38 = extern_kernels.convolution(buf37, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf38, (8, 192, 32, 32), (196608, 1, 6144, 192))
    buf39 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf40 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf42 = empty((192, ), device='cpu', dtype=torch.float32)
    buf43 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_5(c_void_p(buf38.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del primals_10
    # Source Nodes: [x_32], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf43, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (8, 192, 32, 32), (196608, 1, 6144, 192))
    buf45 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf46 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf48 = empty((192, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_6(c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()))
    # Source Nodes: [x_40], Original ATen: [aten.convolution]
    buf49 = extern_kernels.convolution(buf37, primals_121, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 192, 32, 32), (196608, 1, 6144, 192))
    buf50 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf51 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf53 = empty((192, ), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    buf55 = buf54; del buf54  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_relu_7(c_void_p(buf55.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()))
    del primals_12
    del primals_14
    # Source Nodes: [x_46], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (8, 192, 32, 32), (196608, 1, 6144, 192))
    buf57 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf58 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf60 = empty((192, ), device='cpu', dtype=torch.float32)
    buf61 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_8(c_void_p(buf56.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    del primals_16
    # Source Nodes: [x_52], Original ATen: [aten.convolution]
    buf62 = extern_kernels.convolution(buf61, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (8, 192, 32, 32), (196608, 1, 6144, 192))
    buf63 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf64 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf66 = empty((192, ), device='cpu', dtype=torch.float32)
    buf67 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_9(c_void_p(buf62.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    del primals_18
    # Source Nodes: [x_61], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (8, 160, 32, 32), (163840, 1, 5120, 160))
    buf69 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf70 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf72 = empty((160, ), device='cpu', dtype=torch.float32)
    buf73 = empty_strided((8, 160, 32, 32), (163840, 1, 5120, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_10(c_void_p(buf68.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_20
    # Source Nodes: [x_67], Original ATen: [aten.convolution]
    buf74 = extern_kernels.convolution(buf73, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf74, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf75 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf76 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf78 = empty((160, ), device='cpu', dtype=torch.float32)
    buf79 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_11(c_void_p(buf74.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del primals_22
    # Source Nodes: [x_75], Original ATen: [aten.convolution]
    buf80 = extern_kernels.convolution(buf79, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf80, (8, 640, 16, 16), (163840, 1, 10240, 640))
    buf81 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf82 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf84 = empty((640, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_12(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()))
    # Source Nodes: [x_83], Original ATen: [aten.convolution]
    buf85 = extern_kernels.convolution(buf67, primals_127, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf85, (8, 640, 16, 16), (163840, 1, 10240, 640))
    buf86 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf87 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf89 = empty((640, ), device='cpu', dtype=torch.float32)
    buf90 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    buf91 = buf90; del buf90  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_relu_13(c_void_p(buf91.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()))
    del primals_24
    del primals_26
    # Source Nodes: [x_89], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf92, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf93 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf94 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf96 = empty((160, ), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_14(c_void_p(buf92.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_28
    # Source Nodes: [x_95], Original ATen: [aten.convolution]
    buf98 = extern_kernels.convolution(buf97, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf98, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf99 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf100 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf102 = empty((160, ), device='cpu', dtype=torch.float32)
    buf103 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_15(c_void_p(buf98.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    del primals_30
    # Source Nodes: [x_103], Original ATen: [aten.convolution]
    buf104 = extern_kernels.convolution(buf103, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf104, (8, 640, 16, 16), (163840, 1, 10240, 640))
    buf105 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf106 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf108 = empty((640, ), device='cpu', dtype=torch.float32)
    buf109 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_16(c_void_p(buf104.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del primals_32
    # Source Nodes: [x_112], Original ATen: [aten.convolution]
    buf110 = extern_kernels.convolution(buf109, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf110, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf111 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf112 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf114 = empty((160, ), device='cpu', dtype=torch.float32)
    buf115 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_17(c_void_p(buf110.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    del primals_34
    # Source Nodes: [x_118], Original ATen: [aten.convolution]
    buf116 = extern_kernels.convolution(buf115, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf116, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf117 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf118 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf120 = empty((160, ), device='cpu', dtype=torch.float32)
    buf121 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_18(c_void_p(buf116.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    del primals_36
    # Source Nodes: [x_126], Original ATen: [aten.convolution]
    buf122 = extern_kernels.convolution(buf121, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf122, (8, 640, 16, 16), (163840, 1, 10240, 640))
    buf123 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf124 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf126 = empty((640, ), device='cpu', dtype=torch.float32)
    buf127 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_19(c_void_p(buf122.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    del primals_38
    # Source Nodes: [x_135], Original ATen: [aten.convolution]
    buf128 = extern_kernels.convolution(buf127, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf128, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf129 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf130 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf132 = empty((160, ), device='cpu', dtype=torch.float32)
    buf133 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_20(c_void_p(buf128.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()))
    del primals_40
    # Source Nodes: [x_141], Original ATen: [aten.convolution]
    buf134 = extern_kernels.convolution(buf133, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf134, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf135 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf136 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf138 = empty((160, ), device='cpu', dtype=torch.float32)
    buf139 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_21(c_void_p(buf134.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del primals_42
    # Source Nodes: [x_149], Original ATen: [aten.convolution]
    buf140 = extern_kernels.convolution(buf139, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf140, (8, 640, 16, 16), (163840, 1, 10240, 640))
    buf141 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf142 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf144 = empty((640, ), device='cpu', dtype=torch.float32)
    buf145 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_22(c_void_p(buf140.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del primals_44
    # Source Nodes: [x_158], Original ATen: [aten.convolution]
    buf146 = extern_kernels.convolution(buf145, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf146, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf147 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf148 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf150 = empty((160, ), device='cpu', dtype=torch.float32)
    buf151 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_23(c_void_p(buf146.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    del primals_46
    # Source Nodes: [x_164], Original ATen: [aten.convolution]
    buf152 = extern_kernels.convolution(buf151, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf152, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf153 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf154 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf156 = empty((160, ), device='cpu', dtype=torch.float32)
    buf157 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_24(c_void_p(buf152.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()))
    del primals_48
    # Source Nodes: [x_172], Original ATen: [aten.convolution]
    buf158 = extern_kernels.convolution(buf157, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf158, (8, 640, 16, 16), (163840, 1, 10240, 640))
    buf159 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf160 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf162 = empty((640, ), device='cpu', dtype=torch.float32)
    buf163 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_25(c_void_p(buf158.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    del primals_50
    # Source Nodes: [x_181], Original ATen: [aten.convolution]
    buf164 = extern_kernels.convolution(buf163, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf164, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf165 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf166 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf168 = empty((160, ), device='cpu', dtype=torch.float32)
    buf169 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_26(c_void_p(buf164.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    del primals_52
    # Source Nodes: [x_187], Original ATen: [aten.convolution]
    buf170 = extern_kernels.convolution(buf169, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf170, (8, 160, 16, 16), (40960, 1, 2560, 160))
    buf171 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf172 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf174 = empty((160, ), device='cpu', dtype=torch.float32)
    buf175 = empty_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_27(c_void_p(buf170.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    del primals_54
    # Source Nodes: [x_195], Original ATen: [aten.convolution]
    buf176 = extern_kernels.convolution(buf175, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf176, (8, 640, 16, 16), (163840, 1, 10240, 640))
    buf177 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf178 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf180 = empty((640, ), device='cpu', dtype=torch.float32)
    buf181 = empty_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_28(c_void_p(buf176.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del primals_56
    # Source Nodes: [x_204], Original ATen: [aten.convolution]
    buf182 = extern_kernels.convolution(buf181, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf182, (8, 1920, 16, 16), (491520, 1, 30720, 1920))
    buf183 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf184 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf186 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf187 = empty_strided((8, 1920, 16, 16), (491520, 1, 30720, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_29(c_void_p(buf182.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()))
    del primals_58
    # Source Nodes: [x_210], Original ATen: [aten.convolution]
    buf188 = extern_kernels.convolution(buf187, primals_144, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf188, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf189 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf190 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf192 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf193 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_30(c_void_p(buf188.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    del primals_60
    # Source Nodes: [x_218], Original ATen: [aten.convolution]
    buf194 = extern_kernels.convolution(buf193, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf194, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf195 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf196 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf198 = empty((640, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_31(c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()))
    # Source Nodes: [x_226], Original ATen: [aten.convolution]
    buf199 = extern_kernels.convolution(buf181, primals_146, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf199, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf200 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf201 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf203 = empty((640, ), device='cpu', dtype=torch.float32)
    buf204 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    buf205 = buf204; del buf204  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_relu_32(c_void_p(buf205.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf203.data_ptr()))
    del primals_62
    del primals_64
    # Source Nodes: [x_232], Original ATen: [aten.convolution]
    buf206 = extern_kernels.convolution(buf205, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf206, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf207 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf208 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf210 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_33(c_void_p(buf206.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del primals_66
    # Source Nodes: [x_238], Original ATen: [aten.convolution]
    buf212 = extern_kernels.convolution(buf211, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf212, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf213 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf214 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf216 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf217 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_34(c_void_p(buf212.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    del primals_68
    # Source Nodes: [x_246], Original ATen: [aten.convolution]
    buf218 = extern_kernels.convolution(buf217, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf218, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf219 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf220 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf222 = empty((640, ), device='cpu', dtype=torch.float32)
    buf223 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_35(c_void_p(buf218.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    del primals_70
    # Source Nodes: [x_255], Original ATen: [aten.convolution]
    buf224 = extern_kernels.convolution(buf223, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf224, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf225 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf226 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf228 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf229 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_36(c_void_p(buf224.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()))
    del primals_72
    # Source Nodes: [x_261], Original ATen: [aten.convolution]
    buf230 = extern_kernels.convolution(buf229, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf230, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf231 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf232 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf234 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf235 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_37(c_void_p(buf230.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del primals_74
    # Source Nodes: [x_269], Original ATen: [aten.convolution]
    buf236 = extern_kernels.convolution(buf235, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf236, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf237 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf238 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf240 = empty((640, ), device='cpu', dtype=torch.float32)
    buf241 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_38(c_void_p(buf236.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del primals_76
    # Source Nodes: [x_278], Original ATen: [aten.convolution]
    buf242 = extern_kernels.convolution(buf241, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf242, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf243 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf244 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf246 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf247 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_39(c_void_p(buf242.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    del primals_78
    # Source Nodes: [x_284], Original ATen: [aten.convolution]
    buf248 = extern_kernels.convolution(buf247, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf248, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf249 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf250 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf252 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf253 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_40(c_void_p(buf248.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del primals_80
    # Source Nodes: [x_292], Original ATen: [aten.convolution]
    buf254 = extern_kernels.convolution(buf253, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf254, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf255 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf256 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf258 = empty((640, ), device='cpu', dtype=torch.float32)
    buf259 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_41(c_void_p(buf254.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    del primals_82
    # Source Nodes: [x_301], Original ATen: [aten.convolution]
    buf260 = extern_kernels.convolution(buf259, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf260, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf261 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf262 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf264 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf265 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_42(c_void_p(buf260.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del primals_84
    # Source Nodes: [x_307], Original ATen: [aten.convolution]
    buf266 = extern_kernels.convolution(buf265, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf266, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf267 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf268 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf270 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf271 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_43(c_void_p(buf266.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    del primals_86
    # Source Nodes: [x_315], Original ATen: [aten.convolution]
    buf272 = extern_kernels.convolution(buf271, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf272, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf273 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf274 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf276 = empty((640, ), device='cpu', dtype=torch.float32)
    buf277 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_44(c_void_p(buf272.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()))
    del primals_88
    # Source Nodes: [x_324], Original ATen: [aten.convolution]
    buf278 = extern_kernels.convolution(buf277, primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf278, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf279 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf280 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf282 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf283 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_45(c_void_p(buf278.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    del primals_90
    # Source Nodes: [x_330], Original ATen: [aten.convolution]
    buf284 = extern_kernels.convolution(buf283, primals_160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf284, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf285 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf286 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf288 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf289 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_46(c_void_p(buf284.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    del primals_92
    # Source Nodes: [x_338], Original ATen: [aten.convolution]
    buf290 = extern_kernels.convolution(buf289, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf290, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf291 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf292 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf294 = empty((640, ), device='cpu', dtype=torch.float32)
    buf295 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_47(c_void_p(buf290.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    del primals_94
    # Source Nodes: [x_347], Original ATen: [aten.convolution]
    buf296 = extern_kernels.convolution(buf295, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf296, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf297 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf298 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf300 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf301 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_48(c_void_p(buf296.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    del primals_96
    # Source Nodes: [x_353], Original ATen: [aten.convolution]
    buf302 = extern_kernels.convolution(buf301, primals_163, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf302, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf303 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf304 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf306 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf307 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_49(c_void_p(buf302.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()))
    del primals_98
    # Source Nodes: [x_361], Original ATen: [aten.convolution]
    buf308 = extern_kernels.convolution(buf307, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf308, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf309 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf310 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf312 = empty((640, ), device='cpu', dtype=torch.float32)
    buf313 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_50(c_void_p(buf308.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    del primals_100
    # Source Nodes: [x_370], Original ATen: [aten.convolution]
    buf314 = extern_kernels.convolution(buf313, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf314, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf315 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf316 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf318 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf319 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_51(c_void_p(buf314.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    del primals_102
    # Source Nodes: [x_376], Original ATen: [aten.convolution]
    buf320 = extern_kernels.convolution(buf319, primals_166, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf320, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf321 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf322 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf324 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf325 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_52(c_void_p(buf320.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del primals_104
    # Source Nodes: [x_384], Original ATen: [aten.convolution]
    buf326 = extern_kernels.convolution(buf325, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf326, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf327 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf328 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf330 = empty((640, ), device='cpu', dtype=torch.float32)
    buf331 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_53(c_void_p(buf326.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()))
    del primals_106
    # Source Nodes: [x_393], Original ATen: [aten.convolution]
    buf332 = extern_kernels.convolution(buf331, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf332, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf333 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf334 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf336 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf337 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_54(c_void_p(buf332.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    del primals_108
    # Source Nodes: [x_399], Original ATen: [aten.convolution]
    buf338 = extern_kernels.convolution(buf337, primals_169, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf338, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    buf339 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf340 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf342 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf343 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_55(c_void_p(buf338.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    del primals_110
    # Source Nodes: [x_407], Original ATen: [aten.convolution]
    buf344 = extern_kernels.convolution(buf343, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf344, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf345 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf346 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf348 = empty((640, ), device='cpu', dtype=torch.float32)
    buf349 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_56(c_void_p(buf344.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()))
    del primals_112
    # Source Nodes: [x_417], Original ATen: [aten.convolution]
    buf350 = extern_kernels.convolution(buf349, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf350, (8, 2560, 8, 8), (163840, 1, 20480, 2560))
    buf351 = empty_strided((1, 2560, 1, 1), (2560, 1, 2560, 2560), device='cpu', dtype=torch.float32)
    buf352 = empty_strided((1, 2560, 1, 1), (2560, 1, 2560, 2560), device='cpu', dtype=torch.float32)
    buf354 = empty((2560, ), device='cpu', dtype=torch.float32)
    buf355 = empty_strided((8, 2560, 8, 8), (163840, 1, 20480, 2560), device='cpu', dtype=torch.float32)
    buf356 = empty_strided((8, 2560, 1, 1), (2560, 1, 20480, 20480), device='cpu', dtype=torch.float32)
    buf357 = reinterpret_tensor(buf356, (8, 2560), (2560, 1), 0); del buf356  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_view_57(c_void_p(buf357.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    del primals_114
    buf358 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_428], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_173, buf357, reinterpret_tensor(primals_172, (2560, 1000), (1, 2560), 0), alpha=1, beta=1, out=buf358)
    del primals_173
    buf359 = empty_strided((8, 2560, 8, 8), (163840, 1, 20480, 2560), device='cpu', dtype=torch.bool)
    buf366 = reinterpret_tensor(buf16, (32, ), (1, ), 0); del buf16  # reuse
    buf374 = reinterpret_tensor(buf22, (128, ), (1, ), 0); del buf22  # reuse
    buf382 = reinterpret_tensor(buf28, (128, ), (1, ), 0); del buf28  # reuse
    buf390 = reinterpret_tensor(buf33, (128, ), (1, ), 0); del buf33  # reuse
    buf398 = reinterpret_tensor(buf40, (192, ), (1, ), 0); del buf40  # reuse
    buf406 = reinterpret_tensor(buf46, (192, ), (1, ), 0); del buf46  # reuse
    buf414 = reinterpret_tensor(buf51, (192, ), (1, ), 0); del buf51  # reuse
    buf422 = reinterpret_tensor(buf58, (192, ), (1, ), 0); del buf58  # reuse
    buf430 = reinterpret_tensor(buf64, (192, ), (1, ), 0); del buf64  # reuse
    buf438 = reinterpret_tensor(buf70, (160, ), (1, ), 0); del buf70  # reuse
    buf446 = reinterpret_tensor(buf76, (160, ), (1, ), 0); del buf76  # reuse
    buf454 = reinterpret_tensor(buf82, (640, ), (1, ), 0); del buf82  # reuse
    buf462 = reinterpret_tensor(buf87, (640, ), (1, ), 0); del buf87  # reuse
    buf470 = reinterpret_tensor(buf94, (160, ), (1, ), 0); del buf94  # reuse
    buf478 = reinterpret_tensor(buf100, (160, ), (1, ), 0); del buf100  # reuse
    buf486 = reinterpret_tensor(buf106, (640, ), (1, ), 0); del buf106  # reuse
    buf494 = reinterpret_tensor(buf112, (160, ), (1, ), 0); del buf112  # reuse
    buf502 = reinterpret_tensor(buf118, (160, ), (1, ), 0); del buf118  # reuse
    buf510 = reinterpret_tensor(buf124, (640, ), (1, ), 0); del buf124  # reuse
    buf518 = reinterpret_tensor(buf130, (160, ), (1, ), 0); del buf130  # reuse
    buf526 = reinterpret_tensor(buf136, (160, ), (1, ), 0); del buf136  # reuse
    buf534 = reinterpret_tensor(buf142, (640, ), (1, ), 0); del buf142  # reuse
    buf542 = reinterpret_tensor(buf148, (160, ), (1, ), 0); del buf148  # reuse
    buf550 = reinterpret_tensor(buf154, (160, ), (1, ), 0); del buf154  # reuse
    buf558 = reinterpret_tensor(buf160, (640, ), (1, ), 0); del buf160  # reuse
    buf566 = reinterpret_tensor(buf166, (160, ), (1, ), 0); del buf166  # reuse
    buf574 = reinterpret_tensor(buf172, (160, ), (1, ), 0); del buf172  # reuse
    buf582 = reinterpret_tensor(buf178, (640, ), (1, ), 0); del buf178  # reuse
    buf590 = reinterpret_tensor(buf184, (1920, ), (1, ), 0); del buf184  # reuse
    buf598 = reinterpret_tensor(buf190, (1920, ), (1, ), 0); del buf190  # reuse
    buf606 = reinterpret_tensor(buf196, (640, ), (1, ), 0); del buf196  # reuse
    buf614 = reinterpret_tensor(buf201, (640, ), (1, ), 0); del buf201  # reuse
    buf622 = reinterpret_tensor(buf208, (1920, ), (1, ), 0); del buf208  # reuse
    buf630 = reinterpret_tensor(buf214, (1920, ), (1, ), 0); del buf214  # reuse
    buf638 = reinterpret_tensor(buf220, (640, ), (1, ), 0); del buf220  # reuse
    buf646 = reinterpret_tensor(buf226, (1920, ), (1, ), 0); del buf226  # reuse
    buf654 = reinterpret_tensor(buf232, (1920, ), (1, ), 0); del buf232  # reuse
    buf662 = reinterpret_tensor(buf238, (640, ), (1, ), 0); del buf238  # reuse
    buf670 = reinterpret_tensor(buf244, (1920, ), (1, ), 0); del buf244  # reuse
    buf678 = reinterpret_tensor(buf250, (1920, ), (1, ), 0); del buf250  # reuse
    buf686 = reinterpret_tensor(buf256, (640, ), (1, ), 0); del buf256  # reuse
    buf694 = reinterpret_tensor(buf262, (1920, ), (1, ), 0); del buf262  # reuse
    buf702 = reinterpret_tensor(buf268, (1920, ), (1, ), 0); del buf268  # reuse
    buf710 = reinterpret_tensor(buf274, (640, ), (1, ), 0); del buf274  # reuse
    buf718 = reinterpret_tensor(buf280, (1920, ), (1, ), 0); del buf280  # reuse
    buf726 = reinterpret_tensor(buf286, (1920, ), (1, ), 0); del buf286  # reuse
    buf734 = reinterpret_tensor(buf292, (640, ), (1, ), 0); del buf292  # reuse
    buf742 = reinterpret_tensor(buf298, (1920, ), (1, ), 0); del buf298  # reuse
    buf750 = reinterpret_tensor(buf304, (1920, ), (1, ), 0); del buf304  # reuse
    buf758 = reinterpret_tensor(buf310, (640, ), (1, ), 0); del buf310  # reuse
    buf766 = reinterpret_tensor(buf316, (1920, ), (1, ), 0); del buf316  # reuse
    buf774 = reinterpret_tensor(buf322, (1920, ), (1, ), 0); del buf322  # reuse
    buf782 = reinterpret_tensor(buf328, (640, ), (1, ), 0); del buf328  # reuse
    buf790 = reinterpret_tensor(buf334, (1920, ), (1, ), 0); del buf334  # reuse
    buf798 = reinterpret_tensor(buf340, (1920, ), (1, ), 0); del buf340  # reuse
    buf806 = reinterpret_tensor(buf346, (640, ), (1, ), 0); del buf346  # reuse
    buf814 = reinterpret_tensor(buf352, (2560, ), (1, ), 0); del buf352  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_threshold_backward_58(c_void_p(buf366.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(buf750.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(buf766.data_ptr()), c_void_p(buf774.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf798.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()))
    del buf355
    del buf366
    del buf374
    del buf382
    del buf390
    del buf398
    del buf406
    del buf414
    del buf422
    del buf430
    del buf438
    del buf446
    del buf454
    del buf462
    del buf470
    del buf478
    del buf486
    del buf494
    del buf502
    del buf510
    del buf518
    del buf526
    del buf534
    del buf542
    del buf550
    del buf558
    del buf566
    del buf574
    del buf582
    del buf590
    del buf598
    del buf606
    del buf614
    del buf622
    del buf630
    del buf638
    del buf646
    del buf654
    del buf662
    del buf670
    del buf678
    del buf686
    del buf694
    del buf702
    del buf710
    del buf718
    del buf726
    del buf734
    del buf742
    del buf750
    del buf758
    del buf766
    del buf774
    del buf782
    del buf790
    del buf798
    del buf806
    del buf814
    del primals_174
    del primals_175
    del primals_176
    del primals_177
    del primals_178
    del primals_179
    del primals_180
    del primals_181
    del primals_182
    del primals_183
    del primals_184
    del primals_185
    del primals_186
    del primals_187
    del primals_188
    del primals_189
    del primals_190
    del primals_191
    del primals_192
    del primals_193
    del primals_194
    del primals_195
    del primals_196
    del primals_197
    del primals_198
    del primals_199
    del primals_200
    del primals_201
    del primals_202
    del primals_203
    del primals_204
    del primals_205
    del primals_206
    del primals_207
    del primals_208
    del primals_209
    del primals_210
    del primals_211
    del primals_212
    del primals_213
    del primals_214
    del primals_215
    del primals_216
    del primals_217
    del primals_218
    del primals_219
    del primals_220
    del primals_221
    del primals_222
    del primals_223
    del primals_224
    del primals_225
    del primals_226
    del primals_227
    del primals_228
    del primals_229
    del primals_230
    del primals_231
    del primals_232
    del primals_233
    del primals_234
    del primals_235
    del primals_236
    del primals_237
    del primals_238
    del primals_239
    del primals_240
    del primals_241
    del primals_242
    del primals_243
    del primals_244
    del primals_245
    del primals_246
    del primals_247
    del primals_248
    del primals_249
    del primals_250
    del primals_251
    del primals_252
    del primals_253
    del primals_254
    del primals_255
    del primals_256
    del primals_257
    del primals_258
    del primals_259
    del primals_260
    del primals_261
    del primals_262
    del primals_263
    del primals_264
    del primals_265
    del primals_266
    del primals_267
    del primals_268
    del primals_269
    del primals_270
    del primals_271
    del primals_272
    del primals_273
    del primals_274
    del primals_275
    del primals_276
    del primals_277
    del primals_278
    del primals_279
    del primals_280
    del primals_281
    del primals_282
    del primals_283
    del primals_284
    del primals_285
    del primals_286
    del primals_287
    del primals_288
    del primals_289
    del primals_290
    del primals_291
    del primals_292
    del primals_293
    del primals_294
    del primals_295
    del primals_296
    del primals_297
    del primals_298
    del primals_299
    del primals_300
    del primals_301
    del primals_302
    del primals_303
    del primals_304
    del primals_305
    del primals_306
    del primals_307
    del primals_308
    del primals_309
    del primals_310
    del primals_311
    del primals_312
    del primals_313
    del primals_314
    del primals_315
    del primals_316
    del primals_317
    del primals_318
    del primals_319
    del primals_320
    del primals_321
    del primals_322
    del primals_323
    del primals_324
    del primals_325
    del primals_326
    del primals_327
    del primals_328
    del primals_329
    del primals_330
    del primals_331
    del primals_332
    del primals_333
    del primals_334
    del primals_335
    del primals_336
    del primals_337
    del primals_338
    del primals_339
    del primals_340
    del primals_341
    del primals_342
    del primals_343
    del primals_344
    return (buf358, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, buf0, buf1, buf2, primals_118, buf3, buf4, primals_121, buf5, buf6, primals_124, buf7, primals_126, primals_127, primals_128, buf8, primals_130, primals_131, buf9, primals_133, primals_134, buf10, primals_136, primals_137, buf11, primals_139, primals_140, buf12, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, buf13, buf14, buf18, buf19, buf20, buf24, buf25, buf26, buf30, buf31, buf35, buf37, buf38, buf42, buf43, buf44, buf48, buf49, buf53, buf55, buf56, buf60, buf61, buf62, buf66, buf67, buf68, buf72, buf73, buf74, buf78, buf79, buf80, buf84, buf85, buf89, buf91, buf92, buf96, buf97, buf98, buf102, buf103, buf104, buf108, buf109, buf110, buf114, buf115, buf116, buf120, buf121, buf122, buf126, buf127, buf128, buf132, buf133, buf134, buf138, buf139, buf140, buf144, buf145, buf146, buf150, buf151, buf152, buf156, buf157, buf158, buf162, buf163, buf164, buf168, buf169, buf170, buf174, buf175, buf176, buf180, buf181, buf182, buf186, buf187, buf188, buf192, buf193, buf194, buf198, buf199, buf203, buf205, buf206, buf210, buf211, buf212, buf216, buf217, buf218, buf222, buf223, buf224, buf228, buf229, buf230, buf234, buf235, buf236, buf240, buf241, buf242, buf246, buf247, buf248, buf252, buf253, buf254, buf258, buf259, buf260, buf264, buf265, buf266, buf270, buf271, buf272, buf276, buf277, buf278, buf282, buf283, buf284, buf288, buf289, buf290, buf294, buf295, buf296, buf300, buf301, buf302, buf306, buf307, buf308, buf312, buf313, buf314, buf318, buf319, buf320, buf324, buf325, buf326, buf330, buf331, buf332, buf336, buf337, buf338, buf342, buf343, buf344, buf348, buf349, buf350, buf354, buf357, reinterpret_tensor(primals_172, (1000, 2560), (2560, 1), 0), buf359, reinterpret_tensor(buf351, (1, 2560, 1, 1), (2560, 1, 1, 1), 0), reinterpret_tensor(buf345, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf339, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf333, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf327, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf321, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf315, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf309, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf303, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf297, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf291, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf285, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf279, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf273, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf267, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf261, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf255, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf249, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf243, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf237, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf231, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf225, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf219, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf213, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf207, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf200, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf195, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf189, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf183, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf177, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf171, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf165, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf159, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf153, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf147, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf141, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf135, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf129, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf123, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf117, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf111, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf105, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf99, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf93, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf86, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf81, (1, 640, 1, 1), (640, 1, 1, 1), 0), reinterpret_tensor(buf75, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf69, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf63, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf57, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf50, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf45, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf39, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf32, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf27, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf21, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf15, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((192, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((160, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((640, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((1000, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_175 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_178 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_181 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_184 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_187 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_190 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_193 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_196 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_199 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_202 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_205 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_208 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_211 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_214 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_217 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_220 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_223 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_226 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_229 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_232 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_235 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_238 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_241 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_244 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_247 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_250 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_253 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_256 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_259 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_262 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_265 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_268 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_271 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_274 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_277 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_280 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_283 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_286 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_289 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_292 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_295 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_298 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_301 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_304 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_307 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_310 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_313 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_316 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_319 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_322 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_325 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_328 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_331 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_334 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_337 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_340 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_343 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gernet_l', benchmark_compiled_module)
