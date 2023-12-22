
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr10 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr10[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr13[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr13[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
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
    }
}
''')


cpp_fused_max_pool2d_with_indices_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       long* out_ptr1)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(64L + x2 + (128L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(14336L + x2 + (128L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(14400L + x2 + (128L*x1) + (28672L*x0)));
                        auto tmp2 = at::vec::maximum(tmp1, tmp0);
                        auto tmp4 = at::vec::maximum(tmp3, tmp2);
                        auto tmp6 = at::vec::maximum(tmp5, tmp4);
                        tmp6.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (7168L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(112L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (28672L*x1) + (3211264L*x0))];
                            auto tmp1 = in_out_ptr0[static_cast<long>(64L + x3 + (128L*x2) + (28672L*x1) + (3211264L*x0))];
                            auto tmp7 = in_out_ptr0[static_cast<long>(14336L + x3 + (128L*x2) + (28672L*x1) + (3211264L*x0))];
                            auto tmp12 = in_out_ptr0[static_cast<long>(14400L + x3 + (128L*x2) + (28672L*x1) + (3211264L*x0))];
                            auto tmp2 = tmp1 > tmp0;
                            auto tmp3 = c10::convert<long>(1L + (2L*x2) + (448L*x1));
                            auto tmp4 = c10::convert<long>((2L*x2) + (448L*x1));
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp6 = max_propagate_nan(tmp1, tmp0);
                            auto tmp8 = tmp7 > tmp6;
                            auto tmp9 = c10::convert<long>(224L + (2L*x2) + (448L*x1));
                            auto tmp10 = tmp8 ? tmp9 : tmp5;
                            auto tmp11 = max_propagate_nan(tmp7, tmp6);
                            auto tmp13 = tmp12 > tmp11;
                            auto tmp14 = c10::convert<long>(225L + (2L*x2) + (448L*x1));
                            auto tmp15 = tmp13 ? tmp14 : tmp10;
                            auto tmp16 = max_propagate_nan(tmp12, tmp11);
                            out_ptr1[static_cast<long>(x3 + (64L*x2) + (7168L*x1) + (802816L*x0))] = tmp15;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
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
    }
}
''')


cpp_fused_max_pool2d_with_indices_relu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       long* out_ptr1)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(128L + x2 + (256L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(14336L + x2 + (256L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(14464L + x2 + (256L*x1) + (28672L*x0)));
                        auto tmp2 = at::vec::maximum(tmp1, tmp0);
                        auto tmp4 = at::vec::maximum(tmp3, tmp2);
                        auto tmp6 = at::vec::maximum(tmp5, tmp4);
                        tmp6.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (7168L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (256L*x2) + (28672L*x1) + (1605632L*x0))];
                            auto tmp1 = in_out_ptr0[static_cast<long>(128L + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0))];
                            auto tmp7 = in_out_ptr0[static_cast<long>(14336L + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0))];
                            auto tmp12 = in_out_ptr0[static_cast<long>(14464L + x3 + (256L*x2) + (28672L*x1) + (1605632L*x0))];
                            auto tmp2 = tmp1 > tmp0;
                            auto tmp3 = c10::convert<long>(1L + (2L*x2) + (224L*x1));
                            auto tmp4 = c10::convert<long>((2L*x2) + (224L*x1));
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp6 = max_propagate_nan(tmp1, tmp0);
                            auto tmp8 = tmp7 > tmp6;
                            auto tmp9 = c10::convert<long>(112L + (2L*x2) + (224L*x1));
                            auto tmp10 = tmp8 ? tmp9 : tmp5;
                            auto tmp11 = max_propagate_nan(tmp7, tmp6);
                            auto tmp13 = tmp12 > tmp11;
                            auto tmp14 = c10::convert<long>(113L + (2L*x2) + (224L*x1));
                            auto tmp15 = tmp13 ? tmp14 : tmp10;
                            auto tmp16 = max_propagate_nan(tmp12, tmp11);
                            out_ptr1[static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0))] = tmp15;
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
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
    }
}
''')


cpp_fused_max_pool2d_with_indices_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       long* out_ptr1)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(256L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(14336L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(14592L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp2 = at::vec::maximum(tmp1, tmp0);
                        auto tmp4 = at::vec::maximum(tmp3, tmp2);
                        auto tmp6 = at::vec::maximum(tmp5, tmp4);
                        tmp6.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (7168L*x0)));
                    }
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
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                            auto tmp1 = in_out_ptr0[static_cast<long>(256L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                            auto tmp7 = in_out_ptr0[static_cast<long>(14336L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                            auto tmp12 = in_out_ptr0[static_cast<long>(14592L + x3 + (512L*x2) + (28672L*x1) + (802816L*x0))];
                            auto tmp2 = tmp1 > tmp0;
                            auto tmp3 = c10::convert<long>(1L + (2L*x2) + (112L*x1));
                            auto tmp4 = c10::convert<long>((2L*x2) + (112L*x1));
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp6 = max_propagate_nan(tmp1, tmp0);
                            auto tmp8 = tmp7 > tmp6;
                            auto tmp9 = c10::convert<long>(56L + (2L*x2) + (112L*x1));
                            auto tmp10 = tmp8 ? tmp9 : tmp5;
                            auto tmp11 = max_propagate_nan(tmp7, tmp6);
                            auto tmp13 = tmp12 > tmp11;
                            auto tmp14 = c10::convert<long>(57L + (2L*x2) + (112L*x1));
                            auto tmp15 = tmp13 ? tmp14 : tmp10;
                            auto tmp16 = max_propagate_nan(tmp12, tmp11);
                            out_ptr1[static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0))] = tmp15;
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
    }
}
''')


cpp_fused_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
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
    }
}
''')


cpp_fused_max_pool2d_with_indices_relu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       long* out_ptr1)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(14336L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(14848L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp2 = at::vec::maximum(tmp1, tmp0);
                        auto tmp4 = at::vec::maximum(tmp3, tmp2);
                        auto tmp6 = at::vec::maximum(tmp5, tmp4);
                        tmp6.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (7168L*x0)));
                    }
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
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                            auto tmp1 = in_out_ptr0[static_cast<long>(512L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                            auto tmp7 = in_out_ptr0[static_cast<long>(14336L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                            auto tmp12 = in_out_ptr0[static_cast<long>(14848L + x3 + (1024L*x2) + (28672L*x1) + (401408L*x0))];
                            auto tmp2 = tmp1 > tmp0;
                            auto tmp3 = c10::convert<long>(1L + (2L*x2) + (56L*x1));
                            auto tmp4 = c10::convert<long>((2L*x2) + (56L*x1));
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp6 = max_propagate_nan(tmp1, tmp0);
                            auto tmp8 = tmp7 > tmp6;
                            auto tmp9 = c10::convert<long>(28L + (2L*x2) + (56L*x1));
                            auto tmp10 = tmp8 ? tmp9 : tmp5;
                            auto tmp11 = max_propagate_nan(tmp7, tmp6);
                            auto tmp13 = tmp12 > tmp11;
                            auto tmp14 = c10::convert<long>(29L + (2L*x2) + (56L*x1));
                            auto tmp15 = tmp13 ? tmp14 : tmp10;
                            auto tmp16 = max_propagate_nan(tmp12, tmp11);
                            out_ptr1[static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0))] = tmp15;
                        }
                    }
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
    }
}
''')


cpp_fused_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
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
    }
}
''')


cpp_fused__adaptive_avg_pool2d_max_pool2d_with_indices_relu_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       long* out_ptr1,
                       float* out_ptr2)
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
        #pragma omp single
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (14336L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (14336L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(7168L + x2 + (1024L*x1) + (14336L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(7680L + x2 + (1024L*x1) + (14336L*x0)));
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            tmp6.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (3584L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (1024L*x2) + (14336L*x1) + (100352L*x0))];
                                auto tmp1 = in_out_ptr0[static_cast<long>(512L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0))];
                                auto tmp7 = in_out_ptr0[static_cast<long>(7168L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0))];
                                auto tmp12 = in_out_ptr0[static_cast<long>(7680L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0))];
                                auto tmp2 = tmp1 > tmp0;
                                auto tmp3 = c10::convert<long>(1L + (2L*x2) + (28L*x1));
                                auto tmp4 = c10::convert<long>((2L*x2) + (28L*x1));
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = max_propagate_nan(tmp1, tmp0);
                                auto tmp8 = tmp7 > tmp6;
                                auto tmp9 = c10::convert<long>(14L + (2L*x2) + (28L*x1));
                                auto tmp10 = tmp8 ? tmp9 : tmp5;
                                auto tmp11 = max_propagate_nan(tmp7, tmp6);
                                auto tmp13 = tmp12 > tmp11;
                                auto tmp14 = c10::convert<long>(15L + (2L*x2) + (28L*x1));
                                auto tmp15 = tmp13 ? tmp14 : tmp10;
                                auto tmp16 = max_propagate_nan(tmp12, tmp11);
                                out_ptr1[static_cast<long>(x3 + (512L*x2) + (3584L*x1) + (25088L*x0))] = tmp15;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((512L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(49L))) + (25088L*x0) + (c10::div_floor_integer((x1 + x1_inner), 49L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (25088L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = tmp0 * (tmp0>0);
            auto tmp2 = static_cast<float>(0.0);
            auto tmp3 = tmp1 <= tmp2;
            in_out_ptr0[static_cast<long>(x0)] = tmp1;
            out_ptr0[static_cast<long>(x0)] = tmp3;
        }
    }
}
''')


cpp_fused_relu_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       bool* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = tmp0 * (tmp0>0);
            auto tmp2 = static_cast<float>(0.0);
            auto tmp3 = tmp1 <= tmp2;
            in_out_ptr0[static_cast<long>(x0)] = tmp1;
            out_ptr0[static_cast<long>(x0)] = tmp3;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_16, (512, ), (1, ))
    assert_size_stride(primals_17, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (4096, 25088), (25088, 1))
    assert_size_stride(primals_28, (4096, ), (1, ))
    assert_size_stride(primals_29, (4096, 4096), (4096, 1))
    assert_size_stride(primals_30, (4096, ), (1, ))
    assert_size_stride(primals_31, (1000, 4096), (4096, 1))
    assert_size_stride(primals_32, (1000, ), (1, ))
    assert_size_stride(primals_33, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del primals_1
    del primals_11
    del primals_13
    del primals_15
    del primals_17
    del primals_19
    del primals_21
    del primals_23
    del primals_25
    del primals_3
    del primals_33
    del primals_5
    del primals_7
    del primals_9
    # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf13, buf0, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf14, (4, 64, 224, 224), (3211264, 1, 14336, 64))
    del primals_2
    buf15 = buf14; del buf14  # reuse
    cpp_fused_relu_1(c_void_p(buf15.data_ptr()))
    # Source Nodes: [l__mod___features_2], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf15, buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf16, (4, 64, 224, 224), (3211264, 1, 14336, 64))
    del primals_4
    buf17 = buf16; del buf16  # reuse
    buf18 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.int64)
    cpp_fused_max_pool2d_with_indices_relu_2(c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    # Source Nodes: [l__mod___features_5], Original ATen: [aten.convolution]
    buf20 = extern_kernels.convolution(buf18, buf2, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf20, (4, 128, 112, 112), (1605632, 1, 14336, 128))
    del primals_6
    buf21 = buf20; del buf20  # reuse
    cpp_fused_relu_3(c_void_p(buf21.data_ptr()))
    # Source Nodes: [l__mod___features_7], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(buf21, buf3, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf22, (4, 128, 112, 112), (1605632, 1, 14336, 128))
    del primals_8
    buf23 = buf22; del buf22  # reuse
    buf24 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.int64)
    cpp_fused_max_pool2d_with_indices_relu_4(c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    # Source Nodes: [l__mod___features_10], Original ATen: [aten.convolution]
    buf26 = extern_kernels.convolution(buf24, buf4, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf26, (4, 256, 56, 56), (802816, 1, 14336, 256))
    del primals_10
    buf27 = buf26; del buf26  # reuse
    cpp_fused_relu_5(c_void_p(buf27.data_ptr()))
    # Source Nodes: [l__mod___features_12], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(buf27, buf5, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf28, (4, 256, 56, 56), (802816, 1, 14336, 256))
    del primals_12
    buf29 = buf28; del buf28  # reuse
    cpp_fused_relu_6(c_void_p(buf29.data_ptr()))
    # Source Nodes: [l__mod___features_14], Original ATen: [aten.convolution]
    buf30 = extern_kernels.convolution(buf29, buf6, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf30, (4, 256, 56, 56), (802816, 1, 14336, 256))
    del primals_14
    buf31 = buf30; del buf30  # reuse
    buf32 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    buf33 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.int64)
    cpp_fused_max_pool2d_with_indices_relu_7(c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    # Source Nodes: [l__mod___features_17], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf32, buf7, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf34, (4, 512, 28, 28), (401408, 1, 14336, 512))
    del primals_16
    buf35 = buf34; del buf34  # reuse
    cpp_fused_relu_8(c_void_p(buf35.data_ptr()))
    # Source Nodes: [l__mod___features_19], Original ATen: [aten.convolution]
    buf36 = extern_kernels.convolution(buf35, buf8, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf36, (4, 512, 28, 28), (401408, 1, 14336, 512))
    del primals_18
    buf37 = buf36; del buf36  # reuse
    cpp_fused_relu_9(c_void_p(buf37.data_ptr()))
    # Source Nodes: [l__mod___features_21], Original ATen: [aten.convolution]
    buf38 = extern_kernels.convolution(buf37, buf9, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf38, (4, 512, 28, 28), (401408, 1, 14336, 512))
    del primals_20
    buf39 = buf38; del buf38  # reuse
    buf40 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.int64)
    cpp_fused_max_pool2d_with_indices_relu_10(c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    # Source Nodes: [l__mod___features_24], Original ATen: [aten.convolution]
    buf42 = extern_kernels.convolution(buf40, buf10, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf42, (4, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_22
    buf43 = buf42; del buf42  # reuse
    cpp_fused_relu_11(c_void_p(buf43.data_ptr()))
    # Source Nodes: [l__mod___features_26], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf43, buf11, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf44, (4, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_24
    buf45 = buf44; del buf44  # reuse
    cpp_fused_relu_12(c_void_p(buf45.data_ptr()))
    # Source Nodes: [l__mod___features_28], Original ATen: [aten.convolution]
    buf46 = extern_kernels.convolution(buf45, buf12, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf46, (4, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_26
    buf47 = buf46; del buf46  # reuse
    buf48 = empty_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    buf49 = empty_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.int64)
    buf50 = empty((4, 25088), device='cpu', dtype=torch.float32)
    cpp_fused__adaptive_avg_pool2d_max_pool2d_with_indices_relu_view_13(c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    buf51 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_28, buf50, reinterpret_tensor(primals_27, (25088, 4096), (1, 25088), 0), alpha=1, beta=1, out=buf51)
    del primals_28
    buf52 = buf51; del buf51  # reuse
    buf57 = empty((4, 4096), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_14(c_void_p(buf52.data_ptr()), c_void_p(buf57.data_ptr()))
    buf53 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___classifier_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_30, buf52, reinterpret_tensor(primals_29, (4096, 4096), (1, 4096), 0), alpha=1, beta=1, out=buf53)
    del primals_30
    buf54 = buf53; del buf53  # reuse
    buf56 = empty((4, 4096), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_15(c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()))
    buf55 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_32, buf54, reinterpret_tensor(primals_31, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf55)
    del primals_32
    return (buf55, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf15, buf17, buf18, buf19, buf21, buf23, buf24, buf25, buf27, buf29, buf31, buf32, buf33, buf35, buf37, buf39, buf40, buf41, buf43, buf45, buf47, buf48, buf49, buf50, buf52, buf54, reinterpret_tensor(primals_31, (1000, 4096), (4096, 1), 0), buf56, reinterpret_tensor(primals_29, (4096, 4096), (4096, 1), 0), buf57, reinterpret_tensor(primals_27, (4096, 25088), (25088, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((4096, 25088), (25088, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((1000, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vgg16', benchmark_compiled_module)
