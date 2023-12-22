
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
                       float* out_ptr29)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (16L*x1) + (48L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (48L*x0))] = tmp0;
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (4096L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (64L*x2) + (4096L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (4096L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (64L*x2) + (4096L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (4096L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (64L*x2) + (4096L*x0)), static_cast<long>(64L));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (256L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr4 + static_cast<long>(x1 + (64L*x2) + (256L*x0)));
                        }
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (2048L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (128L*x2) + (2048L*x0)), static_cast<long>(128L));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (2048L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (128L*x2) + (2048L*x0)), static_cast<long>(128L));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (2048L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (128L*x2) + (2048L*x0)), static_cast<long>(128L));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (2048L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(x1 + (128L*x2) + (2048L*x0)), static_cast<long>(128L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr9[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (512L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (128L*x2) + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr10[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr11[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr12[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr13[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr14[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr15[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr16[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr17[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr18[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr19[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr20[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr21[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr22[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr23[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr24[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr25[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr26[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr27[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr28[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr28 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
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
                        auto tmp0 = in_ptr29[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr29[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_1 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp4 = out_ptr1[static_cast<long>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(64.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp11.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp15);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = out_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr4[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(64.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr5 + static_cast<long>(x1 + (64L*x0)));
                    tmp19.store(out_ptr6 + static_cast<long>(x1 + (64L*x0)));
                    tmp19.store(out_ptr7 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(64.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp11.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(64.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(64.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(64.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp11.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(64.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(64.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(64.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp11.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(64.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_15 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp4 = out_ptr1[static_cast<long>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(128.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp11.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp15);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = out_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr4[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(128.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr5 + static_cast<long>(x1 + (128L*x0)));
                    tmp19.store(out_ptr6 + static_cast<long>(x1 + (128L*x0)));
                    tmp19.store(out_ptr7 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (128L*x0))];
                        out_ptr0[static_cast<long>(x1 + (2L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp11.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(128.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(128.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (128L*x0))];
                        out_ptr0[static_cast<long>(x1 + (2L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp11.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(128.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(128.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (128L*x0))];
                        out_ptr0[static_cast<long>(x1 + (2L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp11.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(128.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(128.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (128L*x0))];
                        out_ptr0[static_cast<long>(x1 + (2L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp4 = out_ptr1[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp11.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(128.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_37 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp4 = out_ptr1[static_cast<long>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(320.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp15);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = out_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr4[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(320.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr5 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr6 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr7 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(320.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(320.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(320.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(320.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(320.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(320.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(320.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(320.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(320.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(320.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(320.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(320.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(320.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_100 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_105 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(320.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(320.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_110 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_115 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(320.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(320.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_120 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr0[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_125 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(320.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_129 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp4 = out_ptr1[static_cast<long>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(512.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp15);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = out_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr4[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    tmp19.store(out_ptr6 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_134 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_permute_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_native_layer_norm_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x2) + (25088L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
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


cpp_fused_add_detach_native_layer_norm_native_layer_norm_backward_143 = async_compile.cpp('''
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
                       float* in_out_ptr57,
                       float* in_out_ptr58,
                       float* in_out_ptr59,
                       float* in_out_ptr60,
                       float* in_out_ptr61,
                       float* in_out_ptr62,
                       float* in_out_ptr63,
                       float* in_out_ptr64,
                       float* in_out_ptr65,
                       float* in_out_ptr66,
                       float* in_out_ptr67,
                       float* in_out_ptr68,
                       float* in_out_ptr69,
                       float* in_out_ptr70,
                       float* in_out_ptr71,
                       float* in_out_ptr72,
                       float* in_out_ptr73,
                       float* in_out_ptr74,
                       float* in_out_ptr75,
                       float* in_out_ptr76,
                       float* in_out_ptr77,
                       float* in_out_ptr78,
                       float* in_out_ptr79,
                       float* in_out_ptr80,
                       float* in_out_ptr81,
                       float* in_out_ptr82,
                       float* in_out_ptr83,
                       float* in_out_ptr84,
                       float* in_out_ptr85,
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
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
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
                       float* out_ptr24)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr3[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr9 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr10 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr11 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr4[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr12 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr13 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr14 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr5[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr16 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr17 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr6[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr6[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr18 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr19 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr20 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr7[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr7[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr21 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr22 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr23 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr8[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr8[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr24 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr25 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr26 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr9[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr9[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr27 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr28 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr29 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr10[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr10[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr30 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr31 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr32 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr11[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr11[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr33 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr34 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr35 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr12[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr12[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr36 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr37 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr38 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr13[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr13[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr39 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr40 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr41 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr14[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr14[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr42 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr43 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr44 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr15[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr15[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr45 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr46 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr47 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr16[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr16[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr48 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr49 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr50 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr17[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr17[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr51 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr52 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr53 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr18[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr18[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr54 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr55 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr55 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr56 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr19[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr19[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr57 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr57 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr58 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr58 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr59 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr59 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr20[static_cast<long>(x2 + (64L*x1) + (320L*x0))];
                        out_ptr20[static_cast<long>(x1 + (5L*x2) + (320L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr60 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr60 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr61 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr61 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr62 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr62 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr63 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr63 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr21[static_cast<long>(x2 + (64L*x1) + (128L*x0))];
                        out_ptr21[static_cast<long>(x1 + (2L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr64 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr64 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr65 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr65 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr66 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr66 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr22[static_cast<long>(x2 + (64L*x1) + (128L*x0))];
                        out_ptr22[static_cast<long>(x1 + (2L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr67 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr67 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr68 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr68 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr69 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr69 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr23[static_cast<long>(x2 + (64L*x1) + (128L*x0))];
                        out_ptr23[static_cast<long>(x1 + (2L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr70 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr70 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr71 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr71 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr72 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr72 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr24[static_cast<long>(x2 + (64L*x1) + (128L*x0))];
                        out_ptr24[static_cast<long>(x1 + (2L*x2) + (128L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr73 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr73 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr74 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr74 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr75 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr75 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr76 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr76 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr77 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr77 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr78 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr78 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr79 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr79 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr80 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr80 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr81 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr81 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr82 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr82 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr83 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr83 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr84 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr84 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr85 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr85 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64), (64, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (128, 64), (64, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (64, 64), (64, 1))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (512, 64), (64, 1))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (64, 512), (512, 1))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, 64), (64, 1))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, ), (1, ))
    assert_size_stride(primals_33, (128, 64), (64, 1))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (64, 64), (64, 1))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (512, 64), (64, 1))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (64, 512), (512, 1))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, 64), (64, 1))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(primals_48, (64, ), (1, ))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, ), (1, ))
    assert_size_stride(primals_51, (128, 64), (64, 1))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (64, 64), (64, 1))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (512, 64), (64, 1))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_59, (64, 512), (512, 1))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_61, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (128, 128), (128, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (256, 128), (128, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (128, 128), (128, 1))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (1024, 128), (128, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_81, (128, 1024), (1024, 1))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (128, 128), (128, 1))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (128, ), (1, ))
    assert_size_stride(primals_92, (128, ), (1, ))
    assert_size_stride(primals_93, (256, 128), (128, 1))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (128, 128), (128, 1))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_98, (128, ), (1, ))
    assert_size_stride(primals_99, (1024, 128), (128, 1))
    assert_size_stride(primals_100, (1024, ), (1, ))
    assert_size_stride(primals_101, (128, 1024), (1024, 1))
    assert_size_stride(primals_102, (128, ), (1, ))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (128, 128), (128, 1))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (256, 128), (128, 1))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (128, 128), (128, 1))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (1024, 128), (128, 1))
    assert_size_stride(primals_118, (1024, ), (1, ))
    assert_size_stride(primals_119, (128, 1024), (1024, 1))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (128, 128), (128, 1))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (256, 128), (128, 1))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (128, 128), (128, 1))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (1024, 128), (128, 1))
    assert_size_stride(primals_136, (1024, ), (1, ))
    assert_size_stride(primals_137, (128, 1024), (1024, 1))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (320, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(primals_140, (320, ), (1, ))
    assert_size_stride(primals_141, (320, ), (1, ))
    assert_size_stride(primals_142, (320, ), (1, ))
    assert_size_stride(primals_143, (320, ), (1, ))
    assert_size_stride(primals_144, (320, ), (1, ))
    assert_size_stride(primals_145, (320, 320), (320, 1))
    assert_size_stride(primals_146, (320, ), (1, ))
    assert_size_stride(primals_147, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_148, (320, ), (1, ))
    assert_size_stride(primals_149, (320, ), (1, ))
    assert_size_stride(primals_150, (320, ), (1, ))
    assert_size_stride(primals_151, (640, 320), (320, 1))
    assert_size_stride(primals_152, (640, ), (1, ))
    assert_size_stride(primals_153, (320, 320), (320, 1))
    assert_size_stride(primals_154, (320, ), (1, ))
    assert_size_stride(primals_155, (320, ), (1, ))
    assert_size_stride(primals_156, (320, ), (1, ))
    assert_size_stride(primals_157, (1280, 320), (320, 1))
    assert_size_stride(primals_158, (1280, ), (1, ))
    assert_size_stride(primals_159, (320, 1280), (1280, 1))
    assert_size_stride(primals_160, (320, ), (1, ))
    assert_size_stride(primals_161, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_162, (320, ), (1, ))
    assert_size_stride(primals_163, (320, ), (1, ))
    assert_size_stride(primals_164, (320, ), (1, ))
    assert_size_stride(primals_165, (320, 320), (320, 1))
    assert_size_stride(primals_166, (320, ), (1, ))
    assert_size_stride(primals_167, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_168, (320, ), (1, ))
    assert_size_stride(primals_169, (320, ), (1, ))
    assert_size_stride(primals_170, (320, ), (1, ))
    assert_size_stride(primals_171, (640, 320), (320, 1))
    assert_size_stride(primals_172, (640, ), (1, ))
    assert_size_stride(primals_173, (320, 320), (320, 1))
    assert_size_stride(primals_174, (320, ), (1, ))
    assert_size_stride(primals_175, (320, ), (1, ))
    assert_size_stride(primals_176, (320, ), (1, ))
    assert_size_stride(primals_177, (1280, 320), (320, 1))
    assert_size_stride(primals_178, (1280, ), (1, ))
    assert_size_stride(primals_179, (320, 1280), (1280, 1))
    assert_size_stride(primals_180, (320, ), (1, ))
    assert_size_stride(primals_181, (320, ), (1, ))
    assert_size_stride(primals_182, (320, ), (1, ))
    assert_size_stride(primals_183, (320, 320), (320, 1))
    assert_size_stride(primals_184, (320, ), (1, ))
    assert_size_stride(primals_185, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_186, (320, ), (1, ))
    assert_size_stride(primals_187, (320, ), (1, ))
    assert_size_stride(primals_188, (320, ), (1, ))
    assert_size_stride(primals_189, (640, 320), (320, 1))
    assert_size_stride(primals_190, (640, ), (1, ))
    assert_size_stride(primals_191, (320, 320), (320, 1))
    assert_size_stride(primals_192, (320, ), (1, ))
    assert_size_stride(primals_193, (320, ), (1, ))
    assert_size_stride(primals_194, (320, ), (1, ))
    assert_size_stride(primals_195, (1280, 320), (320, 1))
    assert_size_stride(primals_196, (1280, ), (1, ))
    assert_size_stride(primals_197, (320, 1280), (1280, 1))
    assert_size_stride(primals_198, (320, ), (1, ))
    assert_size_stride(primals_199, (320, ), (1, ))
    assert_size_stride(primals_200, (320, ), (1, ))
    assert_size_stride(primals_201, (320, 320), (320, 1))
    assert_size_stride(primals_202, (320, ), (1, ))
    assert_size_stride(primals_203, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_204, (320, ), (1, ))
    assert_size_stride(primals_205, (320, ), (1, ))
    assert_size_stride(primals_206, (320, ), (1, ))
    assert_size_stride(primals_207, (640, 320), (320, 1))
    assert_size_stride(primals_208, (640, ), (1, ))
    assert_size_stride(primals_209, (320, 320), (320, 1))
    assert_size_stride(primals_210, (320, ), (1, ))
    assert_size_stride(primals_211, (320, ), (1, ))
    assert_size_stride(primals_212, (320, ), (1, ))
    assert_size_stride(primals_213, (1280, 320), (320, 1))
    assert_size_stride(primals_214, (1280, ), (1, ))
    assert_size_stride(primals_215, (320, 1280), (1280, 1))
    assert_size_stride(primals_216, (320, ), (1, ))
    assert_size_stride(primals_217, (320, ), (1, ))
    assert_size_stride(primals_218, (320, ), (1, ))
    assert_size_stride(primals_219, (320, 320), (320, 1))
    assert_size_stride(primals_220, (320, ), (1, ))
    assert_size_stride(primals_221, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_222, (320, ), (1, ))
    assert_size_stride(primals_223, (320, ), (1, ))
    assert_size_stride(primals_224, (320, ), (1, ))
    assert_size_stride(primals_225, (640, 320), (320, 1))
    assert_size_stride(primals_226, (640, ), (1, ))
    assert_size_stride(primals_227, (320, 320), (320, 1))
    assert_size_stride(primals_228, (320, ), (1, ))
    assert_size_stride(primals_229, (320, ), (1, ))
    assert_size_stride(primals_230, (320, ), (1, ))
    assert_size_stride(primals_231, (1280, 320), (320, 1))
    assert_size_stride(primals_232, (1280, ), (1, ))
    assert_size_stride(primals_233, (320, 1280), (1280, 1))
    assert_size_stride(primals_234, (320, ), (1, ))
    assert_size_stride(primals_235, (320, ), (1, ))
    assert_size_stride(primals_236, (320, ), (1, ))
    assert_size_stride(primals_237, (320, 320), (320, 1))
    assert_size_stride(primals_238, (320, ), (1, ))
    assert_size_stride(primals_239, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_240, (320, ), (1, ))
    assert_size_stride(primals_241, (320, ), (1, ))
    assert_size_stride(primals_242, (320, ), (1, ))
    assert_size_stride(primals_243, (640, 320), (320, 1))
    assert_size_stride(primals_244, (640, ), (1, ))
    assert_size_stride(primals_245, (320, 320), (320, 1))
    assert_size_stride(primals_246, (320, ), (1, ))
    assert_size_stride(primals_247, (320, ), (1, ))
    assert_size_stride(primals_248, (320, ), (1, ))
    assert_size_stride(primals_249, (1280, 320), (320, 1))
    assert_size_stride(primals_250, (1280, ), (1, ))
    assert_size_stride(primals_251, (320, 1280), (1280, 1))
    assert_size_stride(primals_252, (320, ), (1, ))
    assert_size_stride(primals_253, (320, ), (1, ))
    assert_size_stride(primals_254, (320, ), (1, ))
    assert_size_stride(primals_255, (320, 320), (320, 1))
    assert_size_stride(primals_256, (320, ), (1, ))
    assert_size_stride(primals_257, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_258, (320, ), (1, ))
    assert_size_stride(primals_259, (320, ), (1, ))
    assert_size_stride(primals_260, (320, ), (1, ))
    assert_size_stride(primals_261, (640, 320), (320, 1))
    assert_size_stride(primals_262, (640, ), (1, ))
    assert_size_stride(primals_263, (320, 320), (320, 1))
    assert_size_stride(primals_264, (320, ), (1, ))
    assert_size_stride(primals_265, (320, ), (1, ))
    assert_size_stride(primals_266, (320, ), (1, ))
    assert_size_stride(primals_267, (1280, 320), (320, 1))
    assert_size_stride(primals_268, (1280, ), (1, ))
    assert_size_stride(primals_269, (320, 1280), (1280, 1))
    assert_size_stride(primals_270, (320, ), (1, ))
    assert_size_stride(primals_271, (320, ), (1, ))
    assert_size_stride(primals_272, (320, ), (1, ))
    assert_size_stride(primals_273, (320, 320), (320, 1))
    assert_size_stride(primals_274, (320, ), (1, ))
    assert_size_stride(primals_275, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_276, (320, ), (1, ))
    assert_size_stride(primals_277, (320, ), (1, ))
    assert_size_stride(primals_278, (320, ), (1, ))
    assert_size_stride(primals_279, (640, 320), (320, 1))
    assert_size_stride(primals_280, (640, ), (1, ))
    assert_size_stride(primals_281, (320, 320), (320, 1))
    assert_size_stride(primals_282, (320, ), (1, ))
    assert_size_stride(primals_283, (320, ), (1, ))
    assert_size_stride(primals_284, (320, ), (1, ))
    assert_size_stride(primals_285, (1280, 320), (320, 1))
    assert_size_stride(primals_286, (1280, ), (1, ))
    assert_size_stride(primals_287, (320, 1280), (1280, 1))
    assert_size_stride(primals_288, (320, ), (1, ))
    assert_size_stride(primals_289, (320, ), (1, ))
    assert_size_stride(primals_290, (320, ), (1, ))
    assert_size_stride(primals_291, (320, 320), (320, 1))
    assert_size_stride(primals_292, (320, ), (1, ))
    assert_size_stride(primals_293, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_294, (320, ), (1, ))
    assert_size_stride(primals_295, (320, ), (1, ))
    assert_size_stride(primals_296, (320, ), (1, ))
    assert_size_stride(primals_297, (640, 320), (320, 1))
    assert_size_stride(primals_298, (640, ), (1, ))
    assert_size_stride(primals_299, (320, 320), (320, 1))
    assert_size_stride(primals_300, (320, ), (1, ))
    assert_size_stride(primals_301, (320, ), (1, ))
    assert_size_stride(primals_302, (320, ), (1, ))
    assert_size_stride(primals_303, (1280, 320), (320, 1))
    assert_size_stride(primals_304, (1280, ), (1, ))
    assert_size_stride(primals_305, (320, 1280), (1280, 1))
    assert_size_stride(primals_306, (320, ), (1, ))
    assert_size_stride(primals_307, (320, ), (1, ))
    assert_size_stride(primals_308, (320, ), (1, ))
    assert_size_stride(primals_309, (320, 320), (320, 1))
    assert_size_stride(primals_310, (320, ), (1, ))
    assert_size_stride(primals_311, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_312, (320, ), (1, ))
    assert_size_stride(primals_313, (320, ), (1, ))
    assert_size_stride(primals_314, (320, ), (1, ))
    assert_size_stride(primals_315, (640, 320), (320, 1))
    assert_size_stride(primals_316, (640, ), (1, ))
    assert_size_stride(primals_317, (320, 320), (320, 1))
    assert_size_stride(primals_318, (320, ), (1, ))
    assert_size_stride(primals_319, (320, ), (1, ))
    assert_size_stride(primals_320, (320, ), (1, ))
    assert_size_stride(primals_321, (1280, 320), (320, 1))
    assert_size_stride(primals_322, (1280, ), (1, ))
    assert_size_stride(primals_323, (320, 1280), (1280, 1))
    assert_size_stride(primals_324, (320, ), (1, ))
    assert_size_stride(primals_325, (320, ), (1, ))
    assert_size_stride(primals_326, (320, ), (1, ))
    assert_size_stride(primals_327, (320, 320), (320, 1))
    assert_size_stride(primals_328, (320, ), (1, ))
    assert_size_stride(primals_329, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_330, (320, ), (1, ))
    assert_size_stride(primals_331, (320, ), (1, ))
    assert_size_stride(primals_332, (320, ), (1, ))
    assert_size_stride(primals_333, (640, 320), (320, 1))
    assert_size_stride(primals_334, (640, ), (1, ))
    assert_size_stride(primals_335, (320, 320), (320, 1))
    assert_size_stride(primals_336, (320, ), (1, ))
    assert_size_stride(primals_337, (320, ), (1, ))
    assert_size_stride(primals_338, (320, ), (1, ))
    assert_size_stride(primals_339, (1280, 320), (320, 1))
    assert_size_stride(primals_340, (1280, ), (1, ))
    assert_size_stride(primals_341, (320, 1280), (1280, 1))
    assert_size_stride(primals_342, (320, ), (1, ))
    assert_size_stride(primals_343, (320, ), (1, ))
    assert_size_stride(primals_344, (320, ), (1, ))
    assert_size_stride(primals_345, (320, 320), (320, 1))
    assert_size_stride(primals_346, (320, ), (1, ))
    assert_size_stride(primals_347, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_348, (320, ), (1, ))
    assert_size_stride(primals_349, (320, ), (1, ))
    assert_size_stride(primals_350, (320, ), (1, ))
    assert_size_stride(primals_351, (640, 320), (320, 1))
    assert_size_stride(primals_352, (640, ), (1, ))
    assert_size_stride(primals_353, (320, 320), (320, 1))
    assert_size_stride(primals_354, (320, ), (1, ))
    assert_size_stride(primals_355, (320, ), (1, ))
    assert_size_stride(primals_356, (320, ), (1, ))
    assert_size_stride(primals_357, (1280, 320), (320, 1))
    assert_size_stride(primals_358, (1280, ), (1, ))
    assert_size_stride(primals_359, (320, 1280), (1280, 1))
    assert_size_stride(primals_360, (320, ), (1, ))
    assert_size_stride(primals_361, (320, ), (1, ))
    assert_size_stride(primals_362, (320, ), (1, ))
    assert_size_stride(primals_363, (320, 320), (320, 1))
    assert_size_stride(primals_364, (320, ), (1, ))
    assert_size_stride(primals_365, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_366, (320, ), (1, ))
    assert_size_stride(primals_367, (320, ), (1, ))
    assert_size_stride(primals_368, (320, ), (1, ))
    assert_size_stride(primals_369, (640, 320), (320, 1))
    assert_size_stride(primals_370, (640, ), (1, ))
    assert_size_stride(primals_371, (320, 320), (320, 1))
    assert_size_stride(primals_372, (320, ), (1, ))
    assert_size_stride(primals_373, (320, ), (1, ))
    assert_size_stride(primals_374, (320, ), (1, ))
    assert_size_stride(primals_375, (1280, 320), (320, 1))
    assert_size_stride(primals_376, (1280, ), (1, ))
    assert_size_stride(primals_377, (320, 1280), (1280, 1))
    assert_size_stride(primals_378, (320, ), (1, ))
    assert_size_stride(primals_379, (320, ), (1, ))
    assert_size_stride(primals_380, (320, ), (1, ))
    assert_size_stride(primals_381, (320, 320), (320, 1))
    assert_size_stride(primals_382, (320, ), (1, ))
    assert_size_stride(primals_383, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_384, (320, ), (1, ))
    assert_size_stride(primals_385, (320, ), (1, ))
    assert_size_stride(primals_386, (320, ), (1, ))
    assert_size_stride(primals_387, (640, 320), (320, 1))
    assert_size_stride(primals_388, (640, ), (1, ))
    assert_size_stride(primals_389, (320, 320), (320, 1))
    assert_size_stride(primals_390, (320, ), (1, ))
    assert_size_stride(primals_391, (320, ), (1, ))
    assert_size_stride(primals_392, (320, ), (1, ))
    assert_size_stride(primals_393, (1280, 320), (320, 1))
    assert_size_stride(primals_394, (1280, ), (1, ))
    assert_size_stride(primals_395, (320, 1280), (1280, 1))
    assert_size_stride(primals_396, (320, ), (1, ))
    assert_size_stride(primals_397, (320, ), (1, ))
    assert_size_stride(primals_398, (320, ), (1, ))
    assert_size_stride(primals_399, (320, 320), (320, 1))
    assert_size_stride(primals_400, (320, ), (1, ))
    assert_size_stride(primals_401, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_402, (320, ), (1, ))
    assert_size_stride(primals_403, (320, ), (1, ))
    assert_size_stride(primals_404, (320, ), (1, ))
    assert_size_stride(primals_405, (640, 320), (320, 1))
    assert_size_stride(primals_406, (640, ), (1, ))
    assert_size_stride(primals_407, (320, 320), (320, 1))
    assert_size_stride(primals_408, (320, ), (1, ))
    assert_size_stride(primals_409, (320, ), (1, ))
    assert_size_stride(primals_410, (320, ), (1, ))
    assert_size_stride(primals_411, (1280, 320), (320, 1))
    assert_size_stride(primals_412, (1280, ), (1, ))
    assert_size_stride(primals_413, (320, 1280), (1280, 1))
    assert_size_stride(primals_414, (320, ), (1, ))
    assert_size_stride(primals_415, (320, ), (1, ))
    assert_size_stride(primals_416, (320, ), (1, ))
    assert_size_stride(primals_417, (320, 320), (320, 1))
    assert_size_stride(primals_418, (320, ), (1, ))
    assert_size_stride(primals_419, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_420, (320, ), (1, ))
    assert_size_stride(primals_421, (320, ), (1, ))
    assert_size_stride(primals_422, (320, ), (1, ))
    assert_size_stride(primals_423, (640, 320), (320, 1))
    assert_size_stride(primals_424, (640, ), (1, ))
    assert_size_stride(primals_425, (320, 320), (320, 1))
    assert_size_stride(primals_426, (320, ), (1, ))
    assert_size_stride(primals_427, (320, ), (1, ))
    assert_size_stride(primals_428, (320, ), (1, ))
    assert_size_stride(primals_429, (1280, 320), (320, 1))
    assert_size_stride(primals_430, (1280, ), (1, ))
    assert_size_stride(primals_431, (320, 1280), (1280, 1))
    assert_size_stride(primals_432, (320, ), (1, ))
    assert_size_stride(primals_433, (320, ), (1, ))
    assert_size_stride(primals_434, (320, ), (1, ))
    assert_size_stride(primals_435, (320, 320), (320, 1))
    assert_size_stride(primals_436, (320, ), (1, ))
    assert_size_stride(primals_437, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_438, (320, ), (1, ))
    assert_size_stride(primals_439, (320, ), (1, ))
    assert_size_stride(primals_440, (320, ), (1, ))
    assert_size_stride(primals_441, (640, 320), (320, 1))
    assert_size_stride(primals_442, (640, ), (1, ))
    assert_size_stride(primals_443, (320, 320), (320, 1))
    assert_size_stride(primals_444, (320, ), (1, ))
    assert_size_stride(primals_445, (320, ), (1, ))
    assert_size_stride(primals_446, (320, ), (1, ))
    assert_size_stride(primals_447, (1280, 320), (320, 1))
    assert_size_stride(primals_448, (1280, ), (1, ))
    assert_size_stride(primals_449, (320, 1280), (1280, 1))
    assert_size_stride(primals_450, (320, ), (1, ))
    assert_size_stride(primals_451, (320, ), (1, ))
    assert_size_stride(primals_452, (320, ), (1, ))
    assert_size_stride(primals_453, (320, 320), (320, 1))
    assert_size_stride(primals_454, (320, ), (1, ))
    assert_size_stride(primals_455, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_456, (320, ), (1, ))
    assert_size_stride(primals_457, (320, ), (1, ))
    assert_size_stride(primals_458, (320, ), (1, ))
    assert_size_stride(primals_459, (640, 320), (320, 1))
    assert_size_stride(primals_460, (640, ), (1, ))
    assert_size_stride(primals_461, (320, 320), (320, 1))
    assert_size_stride(primals_462, (320, ), (1, ))
    assert_size_stride(primals_463, (320, ), (1, ))
    assert_size_stride(primals_464, (320, ), (1, ))
    assert_size_stride(primals_465, (1280, 320), (320, 1))
    assert_size_stride(primals_466, (1280, ), (1, ))
    assert_size_stride(primals_467, (320, 1280), (1280, 1))
    assert_size_stride(primals_468, (320, ), (1, ))
    assert_size_stride(primals_469, (512, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_470, (512, ), (1, ))
    assert_size_stride(primals_471, (512, ), (1, ))
    assert_size_stride(primals_472, (512, ), (1, ))
    assert_size_stride(primals_473, (512, ), (1, ))
    assert_size_stride(primals_474, (512, ), (1, ))
    assert_size_stride(primals_475, (512, 512), (512, 1))
    assert_size_stride(primals_476, (512, ), (1, ))
    assert_size_stride(primals_477, (1024, 512), (512, 1))
    assert_size_stride(primals_478, (1024, ), (1, ))
    assert_size_stride(primals_479, (512, 512), (512, 1))
    assert_size_stride(primals_480, (512, ), (1, ))
    assert_size_stride(primals_481, (512, ), (1, ))
    assert_size_stride(primals_482, (512, ), (1, ))
    assert_size_stride(primals_483, (2048, 512), (512, 1))
    assert_size_stride(primals_484, (2048, ), (1, ))
    assert_size_stride(primals_485, (512, 2048), (2048, 1))
    assert_size_stride(primals_486, (512, ), (1, ))
    assert_size_stride(primals_487, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_488, (512, ), (1, ))
    assert_size_stride(primals_489, (512, ), (1, ))
    assert_size_stride(primals_490, (512, ), (1, ))
    assert_size_stride(primals_491, (512, 512), (512, 1))
    assert_size_stride(primals_492, (512, ), (1, ))
    assert_size_stride(primals_493, (1024, 512), (512, 1))
    assert_size_stride(primals_494, (1024, ), (1, ))
    assert_size_stride(primals_495, (512, 512), (512, 1))
    assert_size_stride(primals_496, (512, ), (1, ))
    assert_size_stride(primals_497, (512, ), (1, ))
    assert_size_stride(primals_498, (512, ), (1, ))
    assert_size_stride(primals_499, (2048, 512), (512, 1))
    assert_size_stride(primals_500, (2048, ), (1, ))
    assert_size_stride(primals_501, (512, 2048), (2048, 1))
    assert_size_stride(primals_502, (512, ), (1, ))
    assert_size_stride(primals_503, (512, ), (1, ))
    assert_size_stride(primals_504, (512, ), (1, ))
    assert_size_stride(primals_505, (512, 512), (512, 1))
    assert_size_stride(primals_506, (512, ), (1, ))
    assert_size_stride(primals_507, (1024, 512), (512, 1))
    assert_size_stride(primals_508, (1024, ), (1, ))
    assert_size_stride(primals_509, (512, 512), (512, 1))
    assert_size_stride(primals_510, (512, ), (1, ))
    assert_size_stride(primals_511, (512, ), (1, ))
    assert_size_stride(primals_512, (512, ), (1, ))
    assert_size_stride(primals_513, (2048, 512), (512, 1))
    assert_size_stride(primals_514, (2048, ), (1, ))
    assert_size_stride(primals_515, (512, 2048), (2048, 1))
    assert_size_stride(primals_516, (512, ), (1, ))
    assert_size_stride(primals_517, (512, ), (1, ))
    assert_size_stride(primals_518, (512, ), (1, ))
    assert_size_stride(primals_519, (1000, 512), (512, 1))
    assert_size_stride(primals_520, (1000, ), (1, ))
    assert_size_stride(primals_521, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf26 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf27 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf29 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(primals_401.data_ptr()), c_void_p(primals_419.data_ptr()), c_void_p(primals_437.data_ptr()), c_void_p(primals_455.data_ptr()), c_void_p(primals_469.data_ptr()), c_void_p(primals_521.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    del primals_1
    del primals_107
    del primals_125
    del primals_139
    del primals_147
    del primals_167
    del primals_185
    del primals_203
    del primals_221
    del primals_239
    del primals_257
    del primals_275
    del primals_29
    del primals_293
    del primals_311
    del primals_329
    del primals_347
    del primals_365
    del primals_383
    del primals_401
    del primals_419
    del primals_437
    del primals_455
    del primals_469
    del primals_47
    del primals_521
    del primals_61
    del primals_69
    del primals_89
    del primals_9
    # Source Nodes: [l__mod___patch_embeds_0_proj], Original ATen: [aten.convolution]
    buf30 = extern_kernels.convolution(buf29, buf0, primals_2, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf30, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del primals_2
    buf31 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf32 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf34 = empty((8, 3136, 64), device='cpu', dtype=torch.float32)
    buf35 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf38 = empty((8, 3136, 64), device='cpu', dtype=torch.float32)
    buf39 = empty((25088, 64), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_1(c_void_p(buf30.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf41.data_ptr()))
    del primals_6
    buf40 = reinterpret_tensor(buf30, (25088, 64), (64, 1), 0); del buf30  # reuse
    # Source Nodes: [l__mod___blocks_0_0_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, buf39, reinterpret_tensor(primals_7, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf40)
    del primals_8
    # Source Nodes: [l__mod___blocks_0_0_attn_sr], Original ATen: [aten.convolution]
    buf42 = extern_kernels.convolution(buf41, buf1, primals_10, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf42, (8, 64, 7, 7), (3136, 1, 448, 64))
    del primals_10
    buf43 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf44 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf46 = reinterpret_tensor(buf35, (8, 49, 64), (3136, 64, 1), 0); del buf35  # reuse
    buf47 = reinterpret_tensor(buf31, (392, 64), (64, 1), 0); del buf31  # reuse
    cpp_fused_native_layer_norm_view_2(c_void_p(buf42.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    del primals_12
    buf48 = empty((392, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_0_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_14, buf47, reinterpret_tensor(primals_13, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf48)
    del primals_14
    # Source Nodes: [x_7], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf49 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf40, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf48, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf48, (8, 1, 49, 64), (6272, 0, 128, 1), 64))
    buf50 = buf49[0]
    buf51 = buf49[1]
    buf52 = buf49[2]
    buf53 = buf49[3]
    buf54 = buf49[6]
    buf55 = buf49[7]
    del buf49
    buf57 = empty((25088, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_16, reinterpret_tensor(buf50, (25088, 64), (64, 1), 0), reinterpret_tensor(primals_15, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf57)
    del primals_16
    buf58 = reinterpret_tensor(buf42, (8, 3136, 1), (3136, 1, 25088), 0); del buf42  # reuse
    buf59 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf61 = empty((8, 3136, 64), device='cpu', dtype=torch.float32)
    buf62 = empty((25088, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_3(c_void_p(buf34.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del primals_18
    buf63 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_20, buf62, reinterpret_tensor(primals_19, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf63)
    del primals_20
    buf64 = empty((25088, 512), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_4(c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()))
    buf65 = empty((25088, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_22, buf64, reinterpret_tensor(primals_21, (512, 64), (1, 512), 0), alpha=1, beta=1, out=buf65)
    del primals_22
    buf66 = reinterpret_tensor(buf65, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf65  # reuse
    cpp_fused_view_5(c_void_p(buf66.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf57.data_ptr()))
    del primals_4
    # Source Nodes: [x_20], Original ATen: [aten.convolution]
    buf67 = extern_kernels.convolution(buf66, primals_23, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64)
    assert_size_stride(buf67, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del primals_24
    buf68 = buf58; del buf58  # reuse
    buf69 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf71 = reinterpret_tensor(buf57, (8, 3136, 64), (200704, 64, 1), 0); del buf57  # reuse
    buf72 = empty((25088, 64), device='cpu', dtype=torch.float32)
    buf74 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_6(c_void_p(buf67.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()))
    del primals_26
    buf73 = empty((25088, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_1_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_28, buf72, reinterpret_tensor(primals_27, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf73)
    del primals_28
    # Source Nodes: [l__mod___blocks_0_1_attn_sr], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(buf74, buf2, primals_30, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf75, (8, 64, 7, 7), (3136, 1, 448, 64))
    del primals_30
    buf76 = buf43; del buf43  # reuse
    buf77 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf79 = reinterpret_tensor(buf68, (8, 49, 64), (3136, 64, 1), 0); del buf68  # reuse
    buf80 = empty((392, 64), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_7(c_void_p(buf75.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()))
    del primals_32
    buf81 = empty((392, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_1_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_34, buf80, reinterpret_tensor(primals_33, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf81)
    del primals_34
    # Source Nodes: [x_27], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf82 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf73, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf81, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf81, (8, 1, 49, 64), (6272, 0, 128, 1), 64))
    buf83 = buf82[0]
    buf84 = buf82[1]
    buf85 = buf82[2]
    buf86 = buf82[3]
    buf87 = buf82[6]
    buf88 = buf82[7]
    del buf82
    buf90 = empty((25088, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_36, reinterpret_tensor(buf83, (25088, 64), (64, 1), 0), reinterpret_tensor(primals_35, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf90)
    del primals_36
    buf91 = reinterpret_tensor(buf75, (8, 3136, 1), (3136, 1, 25088), 0); del buf75  # reuse
    buf92 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf94 = empty((8, 3136, 64), device='cpu', dtype=torch.float32)
    buf95 = empty((25088, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_8(c_void_p(buf67.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    del primals_38
    buf96 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_40, buf95, reinterpret_tensor(primals_39, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf96)
    del primals_40
    buf97 = empty((25088, 512), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_9(c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    buf98 = empty((25088, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_42, buf97, reinterpret_tensor(primals_41, (512, 64), (1, 512), 0), alpha=1, beta=1, out=buf98)
    del primals_42
    buf99 = buf91; del buf91  # reuse
    buf100 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf102 = empty((8, 3136, 64), device='cpu', dtype=torch.float32)
    buf103 = empty((25088, 64), device='cpu', dtype=torch.float32)
    buf105 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_10(c_void_p(buf67.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()))
    del primals_44
    buf104 = empty((25088, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_2_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_46, buf103, reinterpret_tensor(primals_45, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf104)
    del primals_46
    # Source Nodes: [l__mod___blocks_0_2_attn_sr], Original ATen: [aten.convolution]
    buf106 = extern_kernels.convolution(buf105, buf3, primals_48, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf106, (8, 64, 7, 7), (3136, 1, 448, 64))
    del primals_48
    buf107 = buf76; del buf76  # reuse
    buf108 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf110 = reinterpret_tensor(buf99, (8, 49, 64), (3136, 64, 1), 0); del buf99  # reuse
    buf111 = empty((392, 64), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_11(c_void_p(buf106.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    del primals_50
    buf112 = empty((392, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_2_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_52, buf111, reinterpret_tensor(primals_51, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf112)
    del primals_52
    # Source Nodes: [x_43], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf113 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf104, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf112, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf112, (8, 1, 49, 64), (6272, 0, 128, 1), 64))
    buf114 = buf113[0]
    buf115 = buf113[1]
    buf116 = buf113[2]
    buf117 = buf113[3]
    buf118 = buf113[6]
    buf119 = buf113[7]
    del buf113
    buf121 = empty((25088, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, reinterpret_tensor(buf114, (25088, 64), (64, 1), 0), reinterpret_tensor(primals_53, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf121)
    del primals_54
    buf122 = reinterpret_tensor(buf121, (8, 3136, 64), (200704, 64, 1), 0); del buf121  # reuse
    buf123 = reinterpret_tensor(buf106, (8, 3136, 1), (3136, 1, 25088), 0); del buf106  # reuse
    buf124 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf126 = empty((8, 3136, 64), device='cpu', dtype=torch.float32)
    buf127 = empty((25088, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_12(c_void_p(buf122.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    del buf123
    del buf67
    del buf90
    del primals_56
    buf128 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_58, buf127, reinterpret_tensor(primals_57, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf128)
    del primals_58
    buf129 = empty((25088, 512), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_13(c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    buf130 = buf98; del buf98  # reuse
    # Source Nodes: [x_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, buf129, reinterpret_tensor(primals_59, (512, 64), (1, 512), 0), alpha=1, beta=1, out=buf130)
    del primals_60
    buf131 = reinterpret_tensor(buf130, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf130  # reuse
    cpp_fused_clone_14(c_void_p(buf131.data_ptr()), c_void_p(buf122.data_ptr()))
    del buf122
    # Source Nodes: [l__mod___patch_embeds_1_proj], Original ATen: [aten.convolution]
    buf132 = extern_kernels.convolution(buf131, buf4, primals_62, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf132, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del primals_62
    buf133 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf134 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf136 = empty((8, 784, 128), device='cpu', dtype=torch.float32)
    buf137 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf138 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf140 = empty((8, 784, 128), device='cpu', dtype=torch.float32)
    buf141 = empty((6272, 128), device='cpu', dtype=torch.float32)
    buf144 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_15(c_void_p(buf132.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf144.data_ptr()))
    del primals_66
    buf142 = reinterpret_tensor(buf132, (6272, 128), (128, 1), 0); del buf132  # reuse
    # Source Nodes: [l__mod___blocks_1_0_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_68, buf141, reinterpret_tensor(primals_67, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf142)
    del primals_68
    buf143 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cpu', dtype=torch.float32)
    cpp_fused_permute_16(c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    # Source Nodes: [l__mod___blocks_1_0_attn_sr], Original ATen: [aten.convolution]
    buf145 = extern_kernels.convolution(buf144, buf5, primals_70, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf145, (8, 128, 7, 7), (6272, 1, 896, 128))
    del primals_70
    buf146 = buf107; del buf107  # reuse
    buf147 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf149 = empty((8, 49, 128), device='cpu', dtype=torch.float32)
    buf150 = empty((392, 128), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_17(c_void_p(buf145.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    del primals_72
    buf151 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_0_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, buf150, reinterpret_tensor(primals_73, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf151)
    del primals_74
    # Source Nodes: [x_64], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf152 = aten._scaled_dot_product_flash_attention(buf143, reinterpret_tensor(buf151, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf151, (8, 2, 49, 64), (12544, 64, 256, 1), 128))
    buf153 = buf152[0]
    buf154 = buf152[1]
    buf155 = buf152[2]
    buf156 = buf152[3]
    buf157 = buf152[6]
    buf158 = buf152[7]
    del buf152
    buf160 = buf142; del buf142  # reuse
    # Source Nodes: [x_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_76, reinterpret_tensor(buf153, (6272, 128), (128, 1), 0), reinterpret_tensor(primals_75, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf160)
    del primals_76
    buf161 = buf137; del buf137  # reuse
    buf162 = buf133; del buf133  # reuse
    buf164 = empty((8, 784, 128), device='cpu', dtype=torch.float32)
    buf165 = empty((6272, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_18(c_void_p(buf136.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del primals_78
    buf166 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_80, buf165, reinterpret_tensor(primals_79, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf166)
    del primals_80
    buf167 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_19(c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    buf168 = empty((6272, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_82, buf167, reinterpret_tensor(primals_81, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf168)
    del primals_82
    buf169 = reinterpret_tensor(buf168, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf168  # reuse
    cpp_fused_view_20(c_void_p(buf169.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf160.data_ptr()))
    del primals_64
    # Source Nodes: [x_77], Original ATen: [aten.convolution]
    buf170 = extern_kernels.convolution(buf169, primals_83, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf170, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del primals_84
    buf171 = buf161; del buf161  # reuse
    buf172 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf174 = reinterpret_tensor(buf160, (8, 784, 128), (100352, 128, 1), 0); del buf160  # reuse
    buf175 = empty((6272, 128), device='cpu', dtype=torch.float32)
    buf178 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_21(c_void_p(buf170.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf178.data_ptr()))
    del primals_86
    buf176 = empty((6272, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_1_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_88, buf175, reinterpret_tensor(primals_87, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf176)
    del primals_88
    buf177 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cpu', dtype=torch.float32)
    cpp_fused_permute_22(c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    # Source Nodes: [l__mod___blocks_1_1_attn_sr], Original ATen: [aten.convolution]
    buf179 = extern_kernels.convolution(buf178, buf6, primals_90, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf179, (8, 128, 7, 7), (6272, 1, 896, 128))
    del primals_90
    buf180 = buf146; del buf146  # reuse
    buf181 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf183 = reinterpret_tensor(buf145, (8, 49, 128), (6272, 128, 1), 0); del buf145  # reuse
    buf184 = empty((392, 128), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_23(c_void_p(buf179.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()))
    del primals_92
    buf185 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_1_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_94, buf184, reinterpret_tensor(primals_93, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf185)
    del primals_94
    # Source Nodes: [x_84], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf186 = aten._scaled_dot_product_flash_attention(buf177, reinterpret_tensor(buf185, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf185, (8, 2, 49, 64), (12544, 64, 256, 1), 128))
    buf187 = buf186[0]
    buf188 = buf186[1]
    buf189 = buf186[2]
    buf190 = buf186[3]
    buf191 = buf186[6]
    buf192 = buf186[7]
    del buf186
    buf194 = buf176; del buf176  # reuse
    # Source Nodes: [x_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_96, reinterpret_tensor(buf187, (6272, 128), (128, 1), 0), reinterpret_tensor(primals_95, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf194)
    del primals_96
    buf195 = buf171; del buf171  # reuse
    buf196 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf198 = empty((8, 784, 128), device='cpu', dtype=torch.float32)
    buf199 = empty((6272, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_24(c_void_p(buf170.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    del primals_98
    buf200 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_100, buf199, reinterpret_tensor(primals_99, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf200)
    del primals_100
    buf201 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_25(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    buf202 = empty((6272, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, buf201, reinterpret_tensor(primals_101, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf202)
    del primals_102
    buf203 = buf195; del buf195  # reuse
    buf204 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf206 = empty((8, 784, 128), device='cpu', dtype=torch.float32)
    buf207 = empty((6272, 128), device='cpu', dtype=torch.float32)
    buf210 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_26(c_void_p(buf170.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf210.data_ptr()))
    del primals_104
    buf208 = empty((6272, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_2_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_106, buf207, reinterpret_tensor(primals_105, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf208)
    del primals_106
    buf209 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cpu', dtype=torch.float32)
    cpp_fused_permute_27(c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    # Source Nodes: [l__mod___blocks_1_2_attn_sr], Original ATen: [aten.convolution]
    buf211 = extern_kernels.convolution(buf210, buf7, primals_108, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf211, (8, 128, 7, 7), (6272, 1, 896, 128))
    del primals_108
    buf212 = buf180; del buf180  # reuse
    buf213 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf215 = reinterpret_tensor(buf179, (8, 49, 128), (6272, 128, 1), 0); del buf179  # reuse
    buf216 = empty((392, 128), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_28(c_void_p(buf211.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del primals_110
    buf217 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_2_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_112, buf216, reinterpret_tensor(primals_111, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf217)
    del primals_112
    # Source Nodes: [x_100], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf218 = aten._scaled_dot_product_flash_attention(buf209, reinterpret_tensor(buf217, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf217, (8, 2, 49, 64), (12544, 64, 256, 1), 128))
    buf219 = buf218[0]
    buf220 = buf218[1]
    buf221 = buf218[2]
    buf222 = buf218[3]
    buf223 = buf218[6]
    buf224 = buf218[7]
    del buf218
    buf226 = buf208; del buf208  # reuse
    # Source Nodes: [x_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_114, reinterpret_tensor(buf219, (6272, 128), (128, 1), 0), reinterpret_tensor(primals_113, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf226)
    del primals_114
    buf227 = reinterpret_tensor(buf226, (8, 784, 128), (100352, 128, 1), 0); del buf226  # reuse
    buf228 = buf203; del buf203  # reuse
    buf229 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf231 = empty((8, 784, 128), device='cpu', dtype=torch.float32)
    buf232 = empty((6272, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_29(c_void_p(buf227.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    del primals_116
    buf233 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_118, buf232, reinterpret_tensor(primals_117, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf233)
    del primals_118
    buf234 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_30(c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()))
    buf235 = buf202; del buf202  # reuse
    # Source Nodes: [x_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf234, reinterpret_tensor(primals_119, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf235)
    del primals_120
    buf236 = buf228; del buf228  # reuse
    buf237 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf239 = reinterpret_tensor(buf194, (8, 784, 128), (100352, 128, 1), 0); del buf194  # reuse
    buf240 = reinterpret_tensor(buf170, (6272, 128), (128, 1), 0); del buf170  # reuse
    buf243 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_31(c_void_p(buf227.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf243.data_ptr()))
    del primals_122
    buf241 = empty((6272, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_3_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_124, buf240, reinterpret_tensor(primals_123, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf241)
    del primals_124
    buf242 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cpu', dtype=torch.float32)
    cpp_fused_permute_32(c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    # Source Nodes: [l__mod___blocks_1_3_attn_sr], Original ATen: [aten.convolution]
    buf244 = extern_kernels.convolution(buf243, buf8, primals_126, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf244, (8, 128, 7, 7), (6272, 1, 896, 128))
    del primals_126
    buf245 = buf212; del buf212  # reuse
    buf246 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf248 = reinterpret_tensor(buf211, (8, 49, 128), (6272, 128, 1), 0); del buf211  # reuse
    buf249 = empty((392, 128), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_33(c_void_p(buf244.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    del buf244
    del primals_128
    buf250 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_3_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf249, reinterpret_tensor(primals_129, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf250)
    del primals_130
    # Source Nodes: [x_116], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf251 = aten._scaled_dot_product_flash_attention(buf242, reinterpret_tensor(buf250, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf250, (8, 2, 49, 64), (12544, 64, 256, 1), 128))
    buf252 = buf251[0]
    buf253 = buf251[1]
    buf254 = buf251[2]
    buf255 = buf251[3]
    buf256 = buf251[6]
    buf257 = buf251[7]
    del buf251
    buf259 = buf241; del buf241  # reuse
    # Source Nodes: [x_118], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, reinterpret_tensor(buf252, (6272, 128), (128, 1), 0), reinterpret_tensor(primals_131, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf259)
    del primals_132
    buf260 = buf236; del buf236  # reuse
    buf261 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf263 = empty((8, 784, 128), device='cpu', dtype=torch.float32)
    buf264 = empty((6272, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_34(c_void_p(buf227.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    del buf260
    del primals_134
    buf265 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_121], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_136, buf264, reinterpret_tensor(primals_135, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf265)
    del primals_136
    buf266 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_35(c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = empty((6272, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_125], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_138, buf266, reinterpret_tensor(primals_137, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf267)
    del primals_138
    buf268 = reinterpret_tensor(buf267, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf267  # reuse
    cpp_fused_clone_36(c_void_p(buf268.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf259.data_ptr()))
    # Source Nodes: [l__mod___patch_embeds_2_proj], Original ATen: [aten.convolution]
    buf269 = extern_kernels.convolution(buf268, buf9, primals_140, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf269, (8, 320, 14, 14), (62720, 1, 4480, 320))
    del primals_140
    buf270 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf271 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf273 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf274 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf275 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf277 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf278 = empty((1568, 320), device='cpu', dtype=torch.float32)
    buf281 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_37(c_void_p(buf269.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf281.data_ptr()))
    del primals_144
    buf279 = reinterpret_tensor(buf269, (1568, 320), (320, 1), 0); del buf269  # reuse
    # Source Nodes: [l__mod___blocks_2_0_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_146, buf278, reinterpret_tensor(primals_145, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf279)
    del primals_146
    buf280 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_38(c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_0_attn_sr], Original ATen: [aten.convolution]
    buf282 = extern_kernels.convolution(buf281, buf10, primals_148, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf282, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_148
    buf283 = buf245; del buf245  # reuse
    buf284 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf286 = empty((8, 49, 320), device='cpu', dtype=torch.float32)
    buf287 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_39(c_void_p(buf282.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    del primals_150
    buf288 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_0_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_152, buf287, reinterpret_tensor(primals_151, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf288)
    del primals_152
    # Source Nodes: [x_137], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf289 = aten._scaled_dot_product_flash_attention(buf280, reinterpret_tensor(buf288, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf288, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf290 = buf289[0]
    buf291 = buf289[1]
    buf292 = buf289[2]
    buf293 = buf289[3]
    buf294 = buf289[6]
    buf295 = buf289[7]
    del buf289
    buf297 = buf279; del buf279  # reuse
    # Source Nodes: [x_139], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_154, reinterpret_tensor(buf290, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_153, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf297)
    del primals_154
    buf298 = buf274; del buf274  # reuse
    buf299 = buf270; del buf270  # reuse
    buf301 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf302 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_40(c_void_p(buf273.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    del primals_156
    buf303 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_158, buf302, reinterpret_tensor(primals_157, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf303)
    del primals_158
    buf304 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_41(c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    buf305 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_160, buf304, reinterpret_tensor(primals_159, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf305)
    del primals_160
    buf306 = reinterpret_tensor(buf305, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf305  # reuse
    cpp_fused_view_42(c_void_p(buf306.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(buf297.data_ptr()))
    del primals_142
    # Source Nodes: [x_150], Original ATen: [aten.convolution]
    buf307 = extern_kernels.convolution(buf306, primals_161, primals_162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320)
    assert_size_stride(buf307, (8, 320, 14, 14), (62720, 1, 4480, 320))
    del primals_162
    buf308 = buf298; del buf298  # reuse
    buf309 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf311 = reinterpret_tensor(buf297, (8, 196, 320), (62720, 320, 1), 0); del buf297  # reuse
    buf312 = empty((1568, 320), device='cpu', dtype=torch.float32)
    buf315 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_43(c_void_p(buf307.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf315.data_ptr()))
    del primals_164
    buf313 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_1_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_166, buf312, reinterpret_tensor(primals_165, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf313)
    del primals_166
    buf314 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_44(c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_1_attn_sr], Original ATen: [aten.convolution]
    buf316 = extern_kernels.convolution(buf315, buf11, primals_168, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf316, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_168
    buf317 = buf283; del buf283  # reuse
    buf318 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf320 = reinterpret_tensor(buf282, (8, 49, 320), (15680, 320, 1), 0); del buf282  # reuse
    buf321 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_45(c_void_p(buf316.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    del primals_170
    buf322 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_1_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_172, buf321, reinterpret_tensor(primals_171, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf322)
    del primals_172
    # Source Nodes: [x_157], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf323 = aten._scaled_dot_product_flash_attention(buf314, reinterpret_tensor(buf322, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf322, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf324 = buf323[0]
    buf325 = buf323[1]
    buf326 = buf323[2]
    buf327 = buf323[3]
    buf328 = buf323[6]
    buf329 = buf323[7]
    del buf323
    buf331 = buf313; del buf313  # reuse
    # Source Nodes: [x_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_174, reinterpret_tensor(buf324, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_173, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf331)
    del primals_174
    buf332 = buf308; del buf308  # reuse
    buf333 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf335 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf336 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_46(c_void_p(buf307.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    del primals_176
    buf337 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_178, buf336, reinterpret_tensor(primals_177, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf337)
    del primals_178
    buf338 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_47(c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()))
    buf339 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_166], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_180, buf338, reinterpret_tensor(primals_179, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf339)
    del primals_180
    buf340 = buf332; del buf332  # reuse
    buf341 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf343 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf344 = empty((1568, 320), device='cpu', dtype=torch.float32)
    buf347 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_48(c_void_p(buf307.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf347.data_ptr()))
    del primals_182
    buf345 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_2_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_184, buf344, reinterpret_tensor(primals_183, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf345)
    del primals_184
    buf346 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_49(c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_2_attn_sr], Original ATen: [aten.convolution]
    buf348 = extern_kernels.convolution(buf347, buf12, primals_186, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf348, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_186
    buf349 = buf317; del buf317  # reuse
    buf350 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf352 = reinterpret_tensor(buf316, (8, 49, 320), (15680, 320, 1), 0); del buf316  # reuse
    buf353 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_50(c_void_p(buf348.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    del primals_188
    buf354 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_2_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_190, buf353, reinterpret_tensor(primals_189, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf354)
    del primals_190
    # Source Nodes: [x_173], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf355 = aten._scaled_dot_product_flash_attention(buf346, reinterpret_tensor(buf354, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf354, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf356 = buf355[0]
    buf357 = buf355[1]
    buf358 = buf355[2]
    buf359 = buf355[3]
    buf360 = buf355[6]
    buf361 = buf355[7]
    del buf355
    buf363 = buf345; del buf345  # reuse
    # Source Nodes: [x_175], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_192, reinterpret_tensor(buf356, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_191, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf363)
    del primals_192
    buf364 = reinterpret_tensor(buf363, (8, 196, 320), (62720, 320, 1), 0); del buf363  # reuse
    buf365 = buf340; del buf340  # reuse
    buf366 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf368 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf369 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_51(c_void_p(buf364.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()))
    del primals_194
    buf370 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_196, buf369, reinterpret_tensor(primals_195, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf370)
    del primals_196
    buf371 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_52(c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    buf372 = buf339; del buf339  # reuse
    # Source Nodes: [x_182], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_198, buf371, reinterpret_tensor(primals_197, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf372)
    del primals_198
    buf373 = buf365; del buf365  # reuse
    buf374 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf376 = reinterpret_tensor(buf331, (8, 196, 320), (62720, 320, 1), 0); del buf331  # reuse
    buf377 = reinterpret_tensor(buf307, (1568, 320), (320, 1), 0); del buf307  # reuse
    buf380 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_53(c_void_p(buf364.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf380.data_ptr()))
    del primals_200
    buf378 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_3_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_202, buf377, reinterpret_tensor(primals_201, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf378)
    del primals_202
    buf379 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_54(c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_3_attn_sr], Original ATen: [aten.convolution]
    buf381 = extern_kernels.convolution(buf380, buf13, primals_204, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf381, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_204
    buf382 = buf349; del buf349  # reuse
    buf383 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf385 = reinterpret_tensor(buf348, (8, 49, 320), (15680, 320, 1), 0); del buf348  # reuse
    buf386 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_55(c_void_p(buf381.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    del primals_206
    buf387 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_3_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_208, buf386, reinterpret_tensor(primals_207, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf387)
    del primals_208
    # Source Nodes: [x_189], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf388 = aten._scaled_dot_product_flash_attention(buf379, reinterpret_tensor(buf387, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf387, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf389 = buf388[0]
    buf390 = buf388[1]
    buf391 = buf388[2]
    buf392 = buf388[3]
    buf393 = buf388[6]
    buf394 = buf388[7]
    del buf388
    buf396 = buf378; del buf378  # reuse
    # Source Nodes: [x_191], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_210, reinterpret_tensor(buf389, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_209, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf396)
    del primals_210
    buf397 = buf373; del buf373  # reuse
    buf398 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf400 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf401 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_56(c_void_p(buf364.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()))
    del primals_212
    buf402 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_194], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_214, buf401, reinterpret_tensor(primals_213, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf402)
    del primals_214
    buf403 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_57(c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()))
    buf404 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_198], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_216, buf403, reinterpret_tensor(primals_215, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf404)
    del primals_216
    buf405 = buf397; del buf397  # reuse
    buf406 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf408 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf409 = empty((1568, 320), device='cpu', dtype=torch.float32)
    buf412 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_58(c_void_p(buf364.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf412.data_ptr()))
    del primals_218
    buf410 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_4_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_220, buf409, reinterpret_tensor(primals_219, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf410)
    del primals_220
    buf411 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_59(c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_4_attn_sr], Original ATen: [aten.convolution]
    buf413 = extern_kernels.convolution(buf412, buf14, primals_222, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf413, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_222
    buf414 = buf382; del buf382  # reuse
    buf415 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf417 = reinterpret_tensor(buf381, (8, 49, 320), (15680, 320, 1), 0); del buf381  # reuse
    buf418 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_60(c_void_p(buf413.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()))
    del primals_224
    buf419 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_4_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_226, buf418, reinterpret_tensor(primals_225, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf419)
    del primals_226
    # Source Nodes: [x_205], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf420 = aten._scaled_dot_product_flash_attention(buf411, reinterpret_tensor(buf419, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf419, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf421 = buf420[0]
    buf422 = buf420[1]
    buf423 = buf420[2]
    buf424 = buf420[3]
    buf425 = buf420[6]
    buf426 = buf420[7]
    del buf420
    buf428 = buf410; del buf410  # reuse
    # Source Nodes: [x_207], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_228, reinterpret_tensor(buf421, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_227, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf428)
    del primals_228
    buf429 = reinterpret_tensor(buf428, (8, 196, 320), (62720, 320, 1), 0); del buf428  # reuse
    buf430 = buf405; del buf405  # reuse
    buf431 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf433 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf434 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_61(c_void_p(buf429.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()))
    del primals_230
    buf435 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_210], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_232, buf434, reinterpret_tensor(primals_231, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf435)
    del primals_232
    buf436 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_62(c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    buf437 = buf404; del buf404  # reuse
    # Source Nodes: [x_214], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_234, buf436, reinterpret_tensor(primals_233, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf437)
    del primals_234
    buf438 = buf430; del buf430  # reuse
    buf439 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf441 = reinterpret_tensor(buf396, (8, 196, 320), (62720, 320, 1), 0); del buf396  # reuse
    buf442 = buf372; del buf372  # reuse
    buf445 = reinterpret_tensor(buf364, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf364  # reuse
    cpp_fused_add_native_layer_norm_view_63(c_void_p(buf429.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf445.data_ptr()))
    del primals_236
    buf443 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_5_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_238, buf442, reinterpret_tensor(primals_237, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf443)
    del primals_238
    buf444 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_64(c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_5_attn_sr], Original ATen: [aten.convolution]
    buf446 = extern_kernels.convolution(buf445, buf15, primals_240, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf446, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_240
    buf447 = buf414; del buf414  # reuse
    buf448 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf450 = reinterpret_tensor(buf413, (8, 49, 320), (15680, 320, 1), 0); del buf413  # reuse
    buf451 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_65(c_void_p(buf446.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()))
    del primals_242
    buf452 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_5_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_244, buf451, reinterpret_tensor(primals_243, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf452)
    del primals_244
    # Source Nodes: [x_221], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf453 = aten._scaled_dot_product_flash_attention(buf444, reinterpret_tensor(buf452, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf452, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf454 = buf453[0]
    buf455 = buf453[1]
    buf456 = buf453[2]
    buf457 = buf453[3]
    buf458 = buf453[6]
    buf459 = buf453[7]
    del buf453
    buf461 = buf443; del buf443  # reuse
    # Source Nodes: [x_223], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_246, reinterpret_tensor(buf454, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_245, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf461)
    del primals_246
    buf462 = buf438; del buf438  # reuse
    buf463 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf465 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf466 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_66(c_void_p(buf429.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()))
    del primals_248
    buf467 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_226], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_250, buf466, reinterpret_tensor(primals_249, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf467)
    del primals_250
    buf468 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_67(c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()))
    buf469 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_230], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_252, buf468, reinterpret_tensor(primals_251, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf469)
    del primals_252
    buf470 = buf462; del buf462  # reuse
    buf471 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf473 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf474 = empty((1568, 320), device='cpu', dtype=torch.float32)
    buf477 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_68(c_void_p(buf429.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf477.data_ptr()))
    del primals_254
    buf475 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_6_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_256, buf474, reinterpret_tensor(primals_255, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf475)
    del primals_256
    buf476 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_69(c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_6_attn_sr], Original ATen: [aten.convolution]
    buf478 = extern_kernels.convolution(buf477, buf16, primals_258, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf478, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_258
    buf479 = buf447; del buf447  # reuse
    buf480 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf482 = reinterpret_tensor(buf446, (8, 49, 320), (15680, 320, 1), 0); del buf446  # reuse
    buf483 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_70(c_void_p(buf478.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()))
    del primals_260
    buf484 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_6_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_262, buf483, reinterpret_tensor(primals_261, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf484)
    del primals_262
    # Source Nodes: [x_237], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf485 = aten._scaled_dot_product_flash_attention(buf476, reinterpret_tensor(buf484, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf484, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf486 = buf485[0]
    buf487 = buf485[1]
    buf488 = buf485[2]
    buf489 = buf485[3]
    buf490 = buf485[6]
    buf491 = buf485[7]
    del buf485
    buf493 = buf475; del buf475  # reuse
    # Source Nodes: [x_239], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_264, reinterpret_tensor(buf486, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_263, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf493)
    del primals_264
    buf494 = reinterpret_tensor(buf493, (8, 196, 320), (62720, 320, 1), 0); del buf493  # reuse
    buf495 = buf470; del buf470  # reuse
    buf496 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf498 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf499 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_71(c_void_p(buf494.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()))
    del primals_266
    buf500 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_242], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_268, buf499, reinterpret_tensor(primals_267, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf500)
    del primals_268
    buf501 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_72(c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()))
    buf502 = buf469; del buf469  # reuse
    # Source Nodes: [x_246], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_270, buf501, reinterpret_tensor(primals_269, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf502)
    del primals_270
    buf503 = buf495; del buf495  # reuse
    buf504 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf506 = reinterpret_tensor(buf461, (8, 196, 320), (62720, 320, 1), 0); del buf461  # reuse
    buf507 = buf437; del buf437  # reuse
    buf510 = reinterpret_tensor(buf429, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf429  # reuse
    cpp_fused_add_native_layer_norm_view_73(c_void_p(buf494.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf510.data_ptr()))
    del primals_272
    buf508 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_7_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_274, buf507, reinterpret_tensor(primals_273, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf508)
    del primals_274
    buf509 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_74(c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_7_attn_sr], Original ATen: [aten.convolution]
    buf511 = extern_kernels.convolution(buf510, buf17, primals_276, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf511, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_276
    buf512 = buf479; del buf479  # reuse
    buf513 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf515 = reinterpret_tensor(buf478, (8, 49, 320), (15680, 320, 1), 0); del buf478  # reuse
    buf516 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_75(c_void_p(buf511.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()))
    del primals_278
    buf517 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_7_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_280, buf516, reinterpret_tensor(primals_279, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf517)
    del primals_280
    # Source Nodes: [x_253], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf518 = aten._scaled_dot_product_flash_attention(buf509, reinterpret_tensor(buf517, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf517, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf519 = buf518[0]
    buf520 = buf518[1]
    buf521 = buf518[2]
    buf522 = buf518[3]
    buf523 = buf518[6]
    buf524 = buf518[7]
    del buf518
    buf526 = buf508; del buf508  # reuse
    # Source Nodes: [x_255], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_282, reinterpret_tensor(buf519, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_281, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf526)
    del primals_282
    buf527 = buf503; del buf503  # reuse
    buf528 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf530 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf531 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_76(c_void_p(buf494.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()))
    del primals_284
    buf532 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_258], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_286, buf531, reinterpret_tensor(primals_285, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf532)
    del primals_286
    buf533 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_77(c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()))
    buf534 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_262], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_288, buf533, reinterpret_tensor(primals_287, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf534)
    del primals_288
    buf535 = buf527; del buf527  # reuse
    buf536 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf538 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf539 = empty((1568, 320), device='cpu', dtype=torch.float32)
    buf542 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_78(c_void_p(buf494.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf542.data_ptr()))
    del primals_290
    buf540 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_8_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_292, buf539, reinterpret_tensor(primals_291, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf540)
    del primals_292
    buf541 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_79(c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_8_attn_sr], Original ATen: [aten.convolution]
    buf543 = extern_kernels.convolution(buf542, buf18, primals_294, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf543, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_294
    buf544 = buf512; del buf512  # reuse
    buf545 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf547 = reinterpret_tensor(buf511, (8, 49, 320), (15680, 320, 1), 0); del buf511  # reuse
    buf548 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_80(c_void_p(buf543.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()))
    del primals_296
    buf549 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_8_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_298, buf548, reinterpret_tensor(primals_297, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf549)
    del primals_298
    # Source Nodes: [x_269], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf550 = aten._scaled_dot_product_flash_attention(buf541, reinterpret_tensor(buf549, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf549, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf551 = buf550[0]
    buf552 = buf550[1]
    buf553 = buf550[2]
    buf554 = buf550[3]
    buf555 = buf550[6]
    buf556 = buf550[7]
    del buf550
    buf558 = buf540; del buf540  # reuse
    # Source Nodes: [x_271], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_300, reinterpret_tensor(buf551, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_299, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf558)
    del primals_300
    buf559 = reinterpret_tensor(buf558, (8, 196, 320), (62720, 320, 1), 0); del buf558  # reuse
    buf560 = buf535; del buf535  # reuse
    buf561 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf563 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf564 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_81(c_void_p(buf559.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()))
    del primals_302
    buf565 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_274], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_304, buf564, reinterpret_tensor(primals_303, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf565)
    del primals_304
    buf566 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_82(c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()))
    buf567 = buf534; del buf534  # reuse
    # Source Nodes: [x_278], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_306, buf566, reinterpret_tensor(primals_305, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf567)
    del primals_306
    buf568 = buf560; del buf560  # reuse
    buf569 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf571 = reinterpret_tensor(buf526, (8, 196, 320), (62720, 320, 1), 0); del buf526  # reuse
    buf572 = buf502; del buf502  # reuse
    buf575 = reinterpret_tensor(buf494, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf494  # reuse
    cpp_fused_add_native_layer_norm_view_83(c_void_p(buf559.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf575.data_ptr()))
    del primals_308
    buf573 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_9_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_310, buf572, reinterpret_tensor(primals_309, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf573)
    del primals_310
    buf574 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_84(c_void_p(buf573.data_ptr()), c_void_p(buf574.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_9_attn_sr], Original ATen: [aten.convolution]
    buf576 = extern_kernels.convolution(buf575, buf19, primals_312, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf576, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_312
    buf577 = buf544; del buf544  # reuse
    buf578 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf580 = reinterpret_tensor(buf543, (8, 49, 320), (15680, 320, 1), 0); del buf543  # reuse
    buf581 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_85(c_void_p(buf576.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()))
    del primals_314
    buf582 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_9_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_316, buf581, reinterpret_tensor(primals_315, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf582)
    del primals_316
    # Source Nodes: [x_285], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf583 = aten._scaled_dot_product_flash_attention(buf574, reinterpret_tensor(buf582, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf582, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf584 = buf583[0]
    buf585 = buf583[1]
    buf586 = buf583[2]
    buf587 = buf583[3]
    buf588 = buf583[6]
    buf589 = buf583[7]
    del buf583
    buf591 = buf573; del buf573  # reuse
    # Source Nodes: [x_287], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_318, reinterpret_tensor(buf584, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_317, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf591)
    del primals_318
    buf592 = buf568; del buf568  # reuse
    buf593 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf595 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf596 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_86(c_void_p(buf559.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf596.data_ptr()))
    del primals_320
    buf597 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_290], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_322, buf596, reinterpret_tensor(primals_321, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf597)
    del primals_322
    buf598 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_87(c_void_p(buf597.data_ptr()), c_void_p(buf598.data_ptr()))
    buf599 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_294], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_324, buf598, reinterpret_tensor(primals_323, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf599)
    del primals_324
    buf600 = buf592; del buf592  # reuse
    buf601 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf603 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf604 = empty((1568, 320), device='cpu', dtype=torch.float32)
    buf607 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_88(c_void_p(buf559.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf607.data_ptr()))
    del primals_326
    buf605 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_10_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_328, buf604, reinterpret_tensor(primals_327, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf605)
    del primals_328
    buf606 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_89(c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_10_attn_sr], Original ATen: [aten.convolution]
    buf608 = extern_kernels.convolution(buf607, buf20, primals_330, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf608, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_330
    buf609 = buf577; del buf577  # reuse
    buf610 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf612 = reinterpret_tensor(buf576, (8, 49, 320), (15680, 320, 1), 0); del buf576  # reuse
    buf613 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_90(c_void_p(buf608.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()))
    del primals_332
    buf614 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_10_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_334, buf613, reinterpret_tensor(primals_333, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf614)
    del primals_334
    # Source Nodes: [x_301], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf615 = aten._scaled_dot_product_flash_attention(buf606, reinterpret_tensor(buf614, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf614, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf616 = buf615[0]
    buf617 = buf615[1]
    buf618 = buf615[2]
    buf619 = buf615[3]
    buf620 = buf615[6]
    buf621 = buf615[7]
    del buf615
    buf623 = buf605; del buf605  # reuse
    # Source Nodes: [x_303], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_336, reinterpret_tensor(buf616, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_335, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf623)
    del primals_336
    buf624 = reinterpret_tensor(buf623, (8, 196, 320), (62720, 320, 1), 0); del buf623  # reuse
    buf625 = buf600; del buf600  # reuse
    buf626 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf628 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf629 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_91(c_void_p(buf624.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf629.data_ptr()))
    del primals_338
    buf630 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_306], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_340, buf629, reinterpret_tensor(primals_339, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf630)
    del primals_340
    buf631 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_92(c_void_p(buf630.data_ptr()), c_void_p(buf631.data_ptr()))
    buf632 = buf599; del buf599  # reuse
    # Source Nodes: [x_310], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_342, buf631, reinterpret_tensor(primals_341, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf632)
    del primals_342
    buf633 = buf625; del buf625  # reuse
    buf634 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf636 = reinterpret_tensor(buf591, (8, 196, 320), (62720, 320, 1), 0); del buf591  # reuse
    buf637 = buf567; del buf567  # reuse
    buf640 = reinterpret_tensor(buf559, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf559  # reuse
    cpp_fused_add_native_layer_norm_view_93(c_void_p(buf624.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf640.data_ptr()))
    del primals_344
    buf638 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_11_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_346, buf637, reinterpret_tensor(primals_345, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf638)
    del primals_346
    buf639 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_94(c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_11_attn_sr], Original ATen: [aten.convolution]
    buf641 = extern_kernels.convolution(buf640, buf21, primals_348, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf641, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_348
    buf642 = buf609; del buf609  # reuse
    buf643 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf645 = reinterpret_tensor(buf608, (8, 49, 320), (15680, 320, 1), 0); del buf608  # reuse
    buf646 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_95(c_void_p(buf641.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf643.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf646.data_ptr()))
    del primals_350
    buf647 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_11_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_352, buf646, reinterpret_tensor(primals_351, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf647)
    del primals_352
    # Source Nodes: [x_317], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf648 = aten._scaled_dot_product_flash_attention(buf639, reinterpret_tensor(buf647, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf647, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf649 = buf648[0]
    buf650 = buf648[1]
    buf651 = buf648[2]
    buf652 = buf648[3]
    buf653 = buf648[6]
    buf654 = buf648[7]
    del buf648
    buf656 = buf638; del buf638  # reuse
    # Source Nodes: [x_319], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_354, reinterpret_tensor(buf649, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_353, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf656)
    del primals_354
    buf657 = buf633; del buf633  # reuse
    buf658 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf660 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf661 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_96(c_void_p(buf624.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf661.data_ptr()))
    del primals_356
    buf662 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_322], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_358, buf661, reinterpret_tensor(primals_357, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf662)
    del primals_358
    buf663 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_97(c_void_p(buf662.data_ptr()), c_void_p(buf663.data_ptr()))
    buf664 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_326], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_360, buf663, reinterpret_tensor(primals_359, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf664)
    del primals_360
    buf665 = buf657; del buf657  # reuse
    buf666 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf668 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf669 = empty((1568, 320), device='cpu', dtype=torch.float32)
    buf672 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_98(c_void_p(buf624.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf672.data_ptr()))
    del primals_362
    buf670 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_12_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_364, buf669, reinterpret_tensor(primals_363, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf670)
    del primals_364
    buf671 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_99(c_void_p(buf670.data_ptr()), c_void_p(buf671.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_12_attn_sr], Original ATen: [aten.convolution]
    buf673 = extern_kernels.convolution(buf672, buf22, primals_366, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf673, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_366
    buf674 = buf642; del buf642  # reuse
    buf675 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf677 = reinterpret_tensor(buf641, (8, 49, 320), (15680, 320, 1), 0); del buf641  # reuse
    buf678 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_100(c_void_p(buf673.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf678.data_ptr()))
    del primals_368
    buf679 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_12_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_370, buf678, reinterpret_tensor(primals_369, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf679)
    del primals_370
    # Source Nodes: [x_333], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf680 = aten._scaled_dot_product_flash_attention(buf671, reinterpret_tensor(buf679, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf679, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf681 = buf680[0]
    buf682 = buf680[1]
    buf683 = buf680[2]
    buf684 = buf680[3]
    buf685 = buf680[6]
    buf686 = buf680[7]
    del buf680
    buf688 = buf670; del buf670  # reuse
    # Source Nodes: [x_335], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_372, reinterpret_tensor(buf681, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_371, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf688)
    del primals_372
    buf689 = reinterpret_tensor(buf688, (8, 196, 320), (62720, 320, 1), 0); del buf688  # reuse
    buf690 = buf665; del buf665  # reuse
    buf691 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf693 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf694 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_101(c_void_p(buf689.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf694.data_ptr()))
    del primals_374
    buf695 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_338], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_376, buf694, reinterpret_tensor(primals_375, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf695)
    del primals_376
    buf696 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_102(c_void_p(buf695.data_ptr()), c_void_p(buf696.data_ptr()))
    buf697 = buf664; del buf664  # reuse
    # Source Nodes: [x_342], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_378, buf696, reinterpret_tensor(primals_377, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf697)
    del primals_378
    buf698 = buf690; del buf690  # reuse
    buf699 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf701 = reinterpret_tensor(buf656, (8, 196, 320), (62720, 320, 1), 0); del buf656  # reuse
    buf702 = buf632; del buf632  # reuse
    buf705 = reinterpret_tensor(buf624, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf624  # reuse
    cpp_fused_add_native_layer_norm_view_103(c_void_p(buf689.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf705.data_ptr()))
    del primals_380
    buf703 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_13_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_382, buf702, reinterpret_tensor(primals_381, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf703)
    del primals_382
    buf704 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_104(c_void_p(buf703.data_ptr()), c_void_p(buf704.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_13_attn_sr], Original ATen: [aten.convolution]
    buf706 = extern_kernels.convolution(buf705, buf23, primals_384, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf706, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_384
    buf707 = buf674; del buf674  # reuse
    buf708 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf710 = reinterpret_tensor(buf673, (8, 49, 320), (15680, 320, 1), 0); del buf673  # reuse
    buf711 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_105(c_void_p(buf706.data_ptr()), c_void_p(primals_385.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf711.data_ptr()))
    del primals_386
    buf712 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_13_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_388, buf711, reinterpret_tensor(primals_387, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf712)
    del primals_388
    # Source Nodes: [x_349], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf713 = aten._scaled_dot_product_flash_attention(buf704, reinterpret_tensor(buf712, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf712, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf714 = buf713[0]
    buf715 = buf713[1]
    buf716 = buf713[2]
    buf717 = buf713[3]
    buf718 = buf713[6]
    buf719 = buf713[7]
    del buf713
    buf721 = buf703; del buf703  # reuse
    # Source Nodes: [x_351], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_390, reinterpret_tensor(buf714, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_389, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf721)
    del primals_390
    buf722 = buf698; del buf698  # reuse
    buf723 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf725 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf726 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_106(c_void_p(buf689.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf726.data_ptr()))
    del primals_392
    buf727 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_354], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_394, buf726, reinterpret_tensor(primals_393, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf727)
    del primals_394
    buf728 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_107(c_void_p(buf727.data_ptr()), c_void_p(buf728.data_ptr()))
    buf729 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_358], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_396, buf728, reinterpret_tensor(primals_395, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf729)
    del primals_396
    buf730 = buf722; del buf722  # reuse
    buf731 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf733 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf734 = empty((1568, 320), device='cpu', dtype=torch.float32)
    buf737 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_108(c_void_p(buf689.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(primals_397.data_ptr()), c_void_p(primals_398.data_ptr()), c_void_p(buf730.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf737.data_ptr()))
    del primals_398
    buf735 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_14_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_400, buf734, reinterpret_tensor(primals_399, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf735)
    del primals_400
    buf736 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_109(c_void_p(buf735.data_ptr()), c_void_p(buf736.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_14_attn_sr], Original ATen: [aten.convolution]
    buf738 = extern_kernels.convolution(buf737, buf24, primals_402, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf738, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_402
    buf739 = buf707; del buf707  # reuse
    buf740 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf742 = reinterpret_tensor(buf706, (8, 49, 320), (15680, 320, 1), 0); del buf706  # reuse
    buf743 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_110(c_void_p(buf738.data_ptr()), c_void_p(primals_403.data_ptr()), c_void_p(primals_404.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(buf743.data_ptr()))
    del primals_404
    buf744 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_14_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_406, buf743, reinterpret_tensor(primals_405, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf744)
    del primals_406
    # Source Nodes: [x_365], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf745 = aten._scaled_dot_product_flash_attention(buf736, reinterpret_tensor(buf744, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf744, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf746 = buf745[0]
    buf747 = buf745[1]
    buf748 = buf745[2]
    buf749 = buf745[3]
    buf750 = buf745[6]
    buf751 = buf745[7]
    del buf745
    buf753 = buf735; del buf735  # reuse
    # Source Nodes: [x_367], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_408, reinterpret_tensor(buf746, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_407, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf753)
    del primals_408
    buf754 = reinterpret_tensor(buf753, (8, 196, 320), (62720, 320, 1), 0); del buf753  # reuse
    buf755 = buf730; del buf730  # reuse
    buf756 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf758 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf759 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_111(c_void_p(buf754.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(primals_409.data_ptr()), c_void_p(primals_410.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf756.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(buf759.data_ptr()))
    del primals_410
    buf760 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_370], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_412, buf759, reinterpret_tensor(primals_411, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf760)
    del primals_412
    buf761 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_112(c_void_p(buf760.data_ptr()), c_void_p(buf761.data_ptr()))
    buf762 = buf729; del buf729  # reuse
    # Source Nodes: [x_374], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_414, buf761, reinterpret_tensor(primals_413, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf762)
    del primals_414
    buf763 = buf755; del buf755  # reuse
    buf764 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf766 = reinterpret_tensor(buf721, (8, 196, 320), (62720, 320, 1), 0); del buf721  # reuse
    buf767 = buf697; del buf697  # reuse
    buf770 = reinterpret_tensor(buf689, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf689  # reuse
    cpp_fused_add_native_layer_norm_view_113(c_void_p(buf754.data_ptr()), c_void_p(buf762.data_ptr()), c_void_p(primals_415.data_ptr()), c_void_p(primals_416.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf764.data_ptr()), c_void_p(buf766.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(buf770.data_ptr()))
    del primals_416
    buf768 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_15_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_418, buf767, reinterpret_tensor(primals_417, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf768)
    del primals_418
    buf769 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_114(c_void_p(buf768.data_ptr()), c_void_p(buf769.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_15_attn_sr], Original ATen: [aten.convolution]
    buf771 = extern_kernels.convolution(buf770, buf25, primals_420, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf771, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_420
    buf772 = buf739; del buf739  # reuse
    buf773 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf775 = reinterpret_tensor(buf738, (8, 49, 320), (15680, 320, 1), 0); del buf738  # reuse
    buf776 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_115(c_void_p(buf771.data_ptr()), c_void_p(primals_421.data_ptr()), c_void_p(primals_422.data_ptr()), c_void_p(buf772.data_ptr()), c_void_p(buf773.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(buf776.data_ptr()))
    del primals_422
    buf777 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_15_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_424, buf776, reinterpret_tensor(primals_423, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf777)
    del primals_424
    # Source Nodes: [x_381], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf778 = aten._scaled_dot_product_flash_attention(buf769, reinterpret_tensor(buf777, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf777, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf779 = buf778[0]
    buf780 = buf778[1]
    buf781 = buf778[2]
    buf782 = buf778[3]
    buf783 = buf778[6]
    buf784 = buf778[7]
    del buf778
    buf786 = buf768; del buf768  # reuse
    # Source Nodes: [x_383], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_426, reinterpret_tensor(buf779, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_425, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf786)
    del primals_426
    buf787 = buf763; del buf763  # reuse
    buf788 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf790 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf791 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_116(c_void_p(buf754.data_ptr()), c_void_p(buf762.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(primals_427.data_ptr()), c_void_p(primals_428.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(buf788.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf791.data_ptr()))
    del primals_428
    buf792 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_386], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_430, buf791, reinterpret_tensor(primals_429, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf792)
    del primals_430
    buf793 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_117(c_void_p(buf792.data_ptr()), c_void_p(buf793.data_ptr()))
    buf794 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_390], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_432, buf793, reinterpret_tensor(primals_431, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf794)
    del primals_432
    buf795 = buf787; del buf787  # reuse
    buf796 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf798 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf799 = empty((1568, 320), device='cpu', dtype=torch.float32)
    buf802 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_118(c_void_p(buf754.data_ptr()), c_void_p(buf762.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf794.data_ptr()), c_void_p(primals_433.data_ptr()), c_void_p(primals_434.data_ptr()), c_void_p(buf795.data_ptr()), c_void_p(buf796.data_ptr()), c_void_p(buf798.data_ptr()), c_void_p(buf799.data_ptr()), c_void_p(buf802.data_ptr()))
    del primals_434
    buf800 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_16_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_436, buf799, reinterpret_tensor(primals_435, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf800)
    del primals_436
    buf801 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_119(c_void_p(buf800.data_ptr()), c_void_p(buf801.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_16_attn_sr], Original ATen: [aten.convolution]
    buf803 = extern_kernels.convolution(buf802, buf26, primals_438, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf803, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_438
    buf804 = buf772; del buf772  # reuse
    buf805 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf807 = reinterpret_tensor(buf771, (8, 49, 320), (15680, 320, 1), 0); del buf771  # reuse
    buf808 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_120(c_void_p(buf803.data_ptr()), c_void_p(primals_439.data_ptr()), c_void_p(primals_440.data_ptr()), c_void_p(buf804.data_ptr()), c_void_p(buf805.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf808.data_ptr()))
    del primals_440
    buf809 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_16_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_442, buf808, reinterpret_tensor(primals_441, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf809)
    del primals_442
    # Source Nodes: [x_397], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf810 = aten._scaled_dot_product_flash_attention(buf801, reinterpret_tensor(buf809, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf809, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf811 = buf810[0]
    buf812 = buf810[1]
    buf813 = buf810[2]
    buf814 = buf810[3]
    buf815 = buf810[6]
    buf816 = buf810[7]
    del buf810
    buf818 = buf800; del buf800  # reuse
    # Source Nodes: [x_399], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_444, reinterpret_tensor(buf811, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_443, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf818)
    del primals_444
    buf819 = reinterpret_tensor(buf818, (8, 196, 320), (62720, 320, 1), 0); del buf818  # reuse
    buf820 = buf795; del buf795  # reuse
    buf821 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf823 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf824 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_121(c_void_p(buf819.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf762.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf794.data_ptr()), c_void_p(primals_445.data_ptr()), c_void_p(primals_446.data_ptr()), c_void_p(buf820.data_ptr()), c_void_p(buf821.data_ptr()), c_void_p(buf823.data_ptr()), c_void_p(buf824.data_ptr()))
    del primals_446
    buf825 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_402], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_448, buf824, reinterpret_tensor(primals_447, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf825)
    del primals_448
    buf826 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_122(c_void_p(buf825.data_ptr()), c_void_p(buf826.data_ptr()))
    buf827 = buf794; del buf794  # reuse
    # Source Nodes: [x_406], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_450, buf826, reinterpret_tensor(primals_449, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf827)
    del primals_450
    buf828 = buf820; del buf820  # reuse
    buf829 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf831 = reinterpret_tensor(buf786, (8, 196, 320), (62720, 320, 1), 0); del buf786  # reuse
    buf832 = buf762; del buf762  # reuse
    buf835 = reinterpret_tensor(buf754, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf754  # reuse
    cpp_fused_add_native_layer_norm_view_123(c_void_p(buf819.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(primals_451.data_ptr()), c_void_p(primals_452.data_ptr()), c_void_p(buf828.data_ptr()), c_void_p(buf829.data_ptr()), c_void_p(buf831.data_ptr()), c_void_p(buf832.data_ptr()), c_void_p(buf835.data_ptr()))
    del primals_452
    buf833 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_17_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_454, buf832, reinterpret_tensor(primals_453, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf833)
    del primals_454
    buf834 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    cpp_fused_permute_124(c_void_p(buf833.data_ptr()), c_void_p(buf834.data_ptr()))
    # Source Nodes: [l__mod___blocks_2_17_attn_sr], Original ATen: [aten.convolution]
    buf836 = extern_kernels.convolution(buf835, buf27, primals_456, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf836, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del primals_456
    buf837 = buf804; del buf804  # reuse
    buf838 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf840 = reinterpret_tensor(buf803, (8, 49, 320), (15680, 320, 1), 0); del buf803  # reuse
    buf841 = empty((392, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_125(c_void_p(buf836.data_ptr()), c_void_p(primals_457.data_ptr()), c_void_p(primals_458.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(buf838.data_ptr()), c_void_p(buf840.data_ptr()), c_void_p(buf841.data_ptr()))
    del buf836
    del primals_458
    buf842 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_17_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_460, buf841, reinterpret_tensor(primals_459, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf842)
    del primals_460
    # Source Nodes: [x_413], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf843 = aten._scaled_dot_product_flash_attention(buf834, reinterpret_tensor(buf842, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf842, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf844 = buf843[0]
    buf845 = buf843[1]
    buf846 = buf843[2]
    buf847 = buf843[3]
    buf848 = buf843[6]
    buf849 = buf843[7]
    del buf843
    buf851 = buf833; del buf833  # reuse
    # Source Nodes: [x_415], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_462, reinterpret_tensor(buf844, (1568, 320), (320, 1), 0), reinterpret_tensor(primals_461, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf851)
    del primals_462
    buf852 = buf828; del buf828  # reuse
    buf853 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf855 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf856 = empty((1568, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_126(c_void_p(buf819.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf851.data_ptr()), c_void_p(primals_463.data_ptr()), c_void_p(primals_464.data_ptr()), c_void_p(buf852.data_ptr()), c_void_p(buf853.data_ptr()), c_void_p(buf855.data_ptr()), c_void_p(buf856.data_ptr()))
    del buf852
    del primals_464
    buf857 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_418], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_466, buf856, reinterpret_tensor(primals_465, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf857)
    del primals_466
    buf858 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_127(c_void_p(buf857.data_ptr()), c_void_p(buf858.data_ptr()))
    buf859 = empty((1568, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_422], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_468, buf858, reinterpret_tensor(primals_467, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf859)
    del primals_468
    buf860 = reinterpret_tensor(buf859, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf859  # reuse
    cpp_fused_clone_128(c_void_p(buf860.data_ptr()), c_void_p(buf819.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf851.data_ptr()))
    # Source Nodes: [l__mod___patch_embeds_3_proj], Original ATen: [aten.convolution]
    buf861 = extern_kernels.convolution(buf860, buf28, primals_470, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf861, (8, 512, 7, 7), (25088, 1, 3584, 512))
    del primals_470
    buf862 = buf837; del buf837  # reuse
    buf863 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf865 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf866 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf867 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf869 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf870 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_129(c_void_p(buf861.data_ptr()), c_void_p(primals_471.data_ptr()), c_void_p(primals_472.data_ptr()), c_void_p(primals_473.data_ptr()), c_void_p(primals_474.data_ptr()), c_void_p(buf862.data_ptr()), c_void_p(buf863.data_ptr()), c_void_p(buf865.data_ptr()), c_void_p(buf866.data_ptr()), c_void_p(buf867.data_ptr()), c_void_p(buf869.data_ptr()), c_void_p(buf870.data_ptr()))
    del primals_474
    buf871 = reinterpret_tensor(buf861, (392, 512), (512, 1), 0); del buf861  # reuse
    # Source Nodes: [l__mod___blocks_3_0_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_476, buf870, reinterpret_tensor(primals_475, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf871)
    del primals_476
    buf872 = empty_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cpu', dtype=torch.float32)
    cpp_fused_permute_130(c_void_p(buf871.data_ptr()), c_void_p(buf872.data_ptr()))
    buf873 = empty((392, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_3_0_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_478, buf870, reinterpret_tensor(primals_477, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf873)
    del primals_478
    # Source Nodes: [x_431], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf874 = aten._scaled_dot_product_flash_attention(buf872, reinterpret_tensor(buf873, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf873, (8, 8, 49, 64), (50176, 64, 1024, 1), 512))
    buf875 = buf874[0]
    buf876 = buf874[1]
    buf877 = buf874[2]
    buf878 = buf874[3]
    buf879 = buf874[6]
    buf880 = buf874[7]
    del buf874
    buf882 = buf871; del buf871  # reuse
    # Source Nodes: [x_433], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_480, reinterpret_tensor(buf875, (392, 512), (512, 1), 0), reinterpret_tensor(primals_479, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf882)
    del primals_480
    buf883 = buf866; del buf866  # reuse
    buf884 = buf862; del buf862  # reuse
    buf886 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf887 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_131(c_void_p(buf865.data_ptr()), c_void_p(primals_471.data_ptr()), c_void_p(primals_472.data_ptr()), c_void_p(buf882.data_ptr()), c_void_p(primals_481.data_ptr()), c_void_p(primals_482.data_ptr()), c_void_p(buf883.data_ptr()), c_void_p(buf884.data_ptr()), c_void_p(buf886.data_ptr()), c_void_p(buf887.data_ptr()))
    del primals_482
    buf888 = reinterpret_tensor(buf259, (392, 2048), (2048, 1), 0); del buf259  # reuse
    # Source Nodes: [x_436], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_484, buf887, reinterpret_tensor(primals_483, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf888)
    del primals_484
    buf889 = reinterpret_tensor(buf235, (392, 2048), (2048, 1), 0); del buf235  # reuse
    cpp_fused_gelu_view_132(c_void_p(buf888.data_ptr()), c_void_p(buf889.data_ptr()))
    buf890 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_440], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_486, buf889, reinterpret_tensor(primals_485, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf890)
    del primals_486
    buf891 = reinterpret_tensor(buf890, (8, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf890  # reuse
    cpp_fused_view_133(c_void_p(buf891.data_ptr()), c_void_p(buf865.data_ptr()), c_void_p(primals_471.data_ptr()), c_void_p(primals_472.data_ptr()), c_void_p(buf882.data_ptr()))
    del primals_472
    # Source Nodes: [x_444], Original ATen: [aten.convolution]
    buf892 = extern_kernels.convolution(buf891, primals_487, primals_488, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf892, (8, 512, 7, 7), (25088, 1, 3584, 512))
    del primals_488
    buf893 = buf883; del buf883  # reuse
    buf894 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf896 = reinterpret_tensor(buf882, (8, 49, 512), (25088, 512, 1), 0); del buf882  # reuse
    buf897 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_134(c_void_p(buf892.data_ptr()), c_void_p(buf891.data_ptr()), c_void_p(primals_489.data_ptr()), c_void_p(primals_490.data_ptr()), c_void_p(buf893.data_ptr()), c_void_p(buf894.data_ptr()), c_void_p(buf896.data_ptr()), c_void_p(buf897.data_ptr()))
    del primals_490
    buf898 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_3_1_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_492, buf897, reinterpret_tensor(primals_491, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf898)
    del primals_492
    buf899 = empty_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cpu', dtype=torch.float32)
    cpp_fused_permute_135(c_void_p(buf898.data_ptr()), c_void_p(buf899.data_ptr()))
    buf900 = empty((392, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_3_1_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_494, buf897, reinterpret_tensor(primals_493, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf900)
    del primals_494
    # Source Nodes: [x_448], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf901 = aten._scaled_dot_product_flash_attention(buf899, reinterpret_tensor(buf900, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf900, (8, 8, 49, 64), (50176, 64, 1024, 1), 512))
    buf902 = buf901[0]
    buf903 = buf901[1]
    buf904 = buf901[2]
    buf905 = buf901[3]
    buf906 = buf901[6]
    buf907 = buf901[7]
    del buf901
    buf909 = buf898; del buf898  # reuse
    # Source Nodes: [x_450], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_496, reinterpret_tensor(buf902, (392, 512), (512, 1), 0), reinterpret_tensor(primals_495, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf909)
    del primals_496
    buf910 = buf893; del buf893  # reuse
    buf911 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf913 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf914 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_136(c_void_p(buf892.data_ptr()), c_void_p(buf891.data_ptr()), c_void_p(buf909.data_ptr()), c_void_p(primals_497.data_ptr()), c_void_p(primals_498.data_ptr()), c_void_p(buf910.data_ptr()), c_void_p(buf911.data_ptr()), c_void_p(buf913.data_ptr()), c_void_p(buf914.data_ptr()))
    del primals_498
    buf915 = reinterpret_tensor(buf227, (392, 2048), (2048, 1), 0); del buf227  # reuse
    # Source Nodes: [x_453], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_500, buf914, reinterpret_tensor(primals_499, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf915)
    del primals_500
    buf916 = empty((392, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_137(c_void_p(buf915.data_ptr()), c_void_p(buf916.data_ptr()))
    buf917 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_457], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_502, buf916, reinterpret_tensor(primals_501, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf917)
    del primals_502
    buf918 = buf910; del buf910  # reuse
    buf919 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf921 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf922 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_138(c_void_p(buf892.data_ptr()), c_void_p(buf891.data_ptr()), c_void_p(buf909.data_ptr()), c_void_p(buf917.data_ptr()), c_void_p(primals_503.data_ptr()), c_void_p(primals_504.data_ptr()), c_void_p(buf918.data_ptr()), c_void_p(buf919.data_ptr()), c_void_p(buf921.data_ptr()), c_void_p(buf922.data_ptr()))
    del primals_504
    buf923 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_3_2_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_506, buf922, reinterpret_tensor(primals_505, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf923)
    del primals_506
    buf924 = empty_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cpu', dtype=torch.float32)
    cpp_fused_permute_139(c_void_p(buf923.data_ptr()), c_void_p(buf924.data_ptr()))
    buf925 = empty((392, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_3_2_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_508, buf922, reinterpret_tensor(primals_507, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf925)
    del primals_508
    # Source Nodes: [x_461], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf926 = aten._scaled_dot_product_flash_attention(buf924, reinterpret_tensor(buf925, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf925, (8, 8, 49, 64), (50176, 64, 1024, 1), 512))
    buf927 = buf926[0]
    buf928 = buf926[1]
    buf929 = buf926[2]
    buf930 = buf926[3]
    buf931 = buf926[6]
    buf932 = buf926[7]
    del buf926
    buf934 = buf923; del buf923  # reuse
    # Source Nodes: [x_463], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_510, reinterpret_tensor(buf927, (392, 512), (512, 1), 0), reinterpret_tensor(primals_509, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf934)
    del primals_510
    buf935 = reinterpret_tensor(buf934, (8, 49, 512), (25088, 512, 1), 0); del buf934  # reuse
    buf936 = buf918; del buf918  # reuse
    buf937 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf939 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf940 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_140(c_void_p(buf935.data_ptr()), c_void_p(buf892.data_ptr()), c_void_p(buf891.data_ptr()), c_void_p(buf909.data_ptr()), c_void_p(buf917.data_ptr()), c_void_p(primals_511.data_ptr()), c_void_p(primals_512.data_ptr()), c_void_p(buf936.data_ptr()), c_void_p(buf937.data_ptr()), c_void_p(buf939.data_ptr()), c_void_p(buf940.data_ptr()))
    del primals_512
    buf941 = empty((392, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_466], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_514, buf940, reinterpret_tensor(primals_513, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf941)
    del primals_514
    buf942 = empty((392, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_141(c_void_p(buf941.data_ptr()), c_void_p(buf942.data_ptr()))
    buf943 = buf917; del buf917  # reuse
    # Source Nodes: [x_470], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_516, buf942, reinterpret_tensor(primals_515, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf943)
    del primals_516
    buf944 = buf936; del buf936  # reuse
    buf945 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf947 = reinterpret_tensor(buf909, (8, 49, 512), (25088, 512, 1), 0); del buf909  # reuse
    buf948 = empty((8, 512), device='cpu', dtype=torch.float32)
    buf949 = buf948; del buf948  # reuse
    cpp_fused_add_mean_native_layer_norm_142(c_void_p(buf949.data_ptr()), c_void_p(buf935.data_ptr()), c_void_p(buf943.data_ptr()), c_void_p(primals_517.data_ptr()), c_void_p(primals_518.data_ptr()), c_void_p(buf944.data_ptr()), c_void_p(buf945.data_ptr()), c_void_p(buf947.data_ptr()))
    del buf944
    del primals_518
    buf950 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_520, buf949, reinterpret_tensor(primals_519, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf950)
    del primals_520
    buf951 = reinterpret_tensor(buf945, (8, 49, 1), (49, 1, 1), 0); del buf945  # reuse
    buf952 = reinterpret_tensor(buf937, (8, 49, 1), (49, 1, 1), 0); del buf937  # reuse
    buf953 = reinterpret_tensor(buf943, (8, 8, 49, 64), (25088, 1, 512, 8), 0); del buf943  # reuse
    buf954 = reinterpret_tensor(buf919, (8, 49, 1), (49, 1, 1), 0); del buf919  # reuse
    buf955 = reinterpret_tensor(buf911, (8, 49, 1), (49, 1, 1), 0); del buf911  # reuse
    buf956 = reinterpret_tensor(buf935, (8, 8, 49, 64), (25088, 1, 512, 8), 0); del buf935  # reuse
    buf957 = reinterpret_tensor(buf894, (8, 49, 1), (49, 1, 1), 0); del buf894  # reuse
    buf958 = reinterpret_tensor(buf884, (8, 49, 1), (49, 1, 1), 0); del buf884  # reuse
    buf959 = reinterpret_tensor(buf892, (8, 8, 49, 64), (25088, 1, 512, 8), 0); del buf892  # reuse
    buf960 = reinterpret_tensor(buf867, (8, 49, 1), (49, 1, 1), 0); del buf867  # reuse
    buf961 = reinterpret_tensor(buf863, (8, 49, 1), (49, 1, 1), 0); del buf863  # reuse
    buf962 = reinterpret_tensor(buf853, (8, 196, 1), (196, 1, 1), 0); del buf853  # reuse
    buf963 = reinterpret_tensor(buf851, (8, 5, 196, 64), (62720, 1, 320, 5), 0); del buf851  # reuse
    buf964 = reinterpret_tensor(buf838, (8, 49, 1), (49, 1, 1), 0); del buf838  # reuse
    buf965 = reinterpret_tensor(buf829, (8, 196, 1), (196, 1, 1), 0); del buf829  # reuse
    buf966 = reinterpret_tensor(buf821, (8, 196, 1), (196, 1, 1), 0); del buf821  # reuse
    buf967 = reinterpret_tensor(buf827, (8, 5, 196, 64), (62720, 1, 320, 5), 0); del buf827  # reuse
    buf968 = reinterpret_tensor(buf805, (8, 49, 1), (49, 1, 1), 0); del buf805  # reuse
    buf969 = reinterpret_tensor(buf796, (8, 196, 1), (196, 1, 1), 0); del buf796  # reuse
    buf970 = reinterpret_tensor(buf788, (8, 196, 1), (196, 1, 1), 0); del buf788  # reuse
    buf971 = reinterpret_tensor(buf819, (8, 5, 196, 64), (62720, 1, 320, 5), 0); del buf819  # reuse
    buf972 = reinterpret_tensor(buf773, (8, 49, 1), (49, 1, 1), 0); del buf773  # reuse
    buf973 = reinterpret_tensor(buf764, (8, 196, 1), (196, 1, 1), 0); del buf764  # reuse
    buf974 = reinterpret_tensor(buf756, (8, 196, 1), (196, 1, 1), 0); del buf756  # reuse
    buf975 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf976 = reinterpret_tensor(buf740, (8, 49, 1), (49, 1, 1), 0); del buf740  # reuse
    buf977 = reinterpret_tensor(buf731, (8, 196, 1), (196, 1, 1), 0); del buf731  # reuse
    buf978 = reinterpret_tensor(buf723, (8, 196, 1), (196, 1, 1), 0); del buf723  # reuse
    buf979 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf980 = reinterpret_tensor(buf708, (8, 49, 1), (49, 1, 1), 0); del buf708  # reuse
    buf981 = reinterpret_tensor(buf699, (8, 196, 1), (196, 1, 1), 0); del buf699  # reuse
    buf982 = reinterpret_tensor(buf691, (8, 196, 1), (196, 1, 1), 0); del buf691  # reuse
    buf983 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf984 = reinterpret_tensor(buf675, (8, 49, 1), (49, 1, 1), 0); del buf675  # reuse
    buf985 = reinterpret_tensor(buf666, (8, 196, 1), (196, 1, 1), 0); del buf666  # reuse
    buf986 = reinterpret_tensor(buf658, (8, 196, 1), (196, 1, 1), 0); del buf658  # reuse
    buf987 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf988 = reinterpret_tensor(buf643, (8, 49, 1), (49, 1, 1), 0); del buf643  # reuse
    buf989 = reinterpret_tensor(buf634, (8, 196, 1), (196, 1, 1), 0); del buf634  # reuse
    buf990 = reinterpret_tensor(buf626, (8, 196, 1), (196, 1, 1), 0); del buf626  # reuse
    buf991 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf992 = reinterpret_tensor(buf610, (8, 49, 1), (49, 1, 1), 0); del buf610  # reuse
    buf993 = reinterpret_tensor(buf601, (8, 196, 1), (196, 1, 1), 0); del buf601  # reuse
    buf994 = reinterpret_tensor(buf593, (8, 196, 1), (196, 1, 1), 0); del buf593  # reuse
    buf995 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf996 = reinterpret_tensor(buf578, (8, 49, 1), (49, 1, 1), 0); del buf578  # reuse
    buf997 = reinterpret_tensor(buf569, (8, 196, 1), (196, 1, 1), 0); del buf569  # reuse
    buf998 = reinterpret_tensor(buf561, (8, 196, 1), (196, 1, 1), 0); del buf561  # reuse
    buf999 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf1000 = reinterpret_tensor(buf545, (8, 49, 1), (49, 1, 1), 0); del buf545  # reuse
    buf1001 = reinterpret_tensor(buf536, (8, 196, 1), (196, 1, 1), 0); del buf536  # reuse
    buf1002 = reinterpret_tensor(buf528, (8, 196, 1), (196, 1, 1), 0); del buf528  # reuse
    buf1003 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf1004 = reinterpret_tensor(buf513, (8, 49, 1), (49, 1, 1), 0); del buf513  # reuse
    buf1005 = reinterpret_tensor(buf504, (8, 196, 1), (196, 1, 1), 0); del buf504  # reuse
    buf1006 = reinterpret_tensor(buf496, (8, 196, 1), (196, 1, 1), 0); del buf496  # reuse
    buf1007 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf1008 = reinterpret_tensor(buf480, (8, 49, 1), (49, 1, 1), 0); del buf480  # reuse
    buf1009 = reinterpret_tensor(buf471, (8, 196, 1), (196, 1, 1), 0); del buf471  # reuse
    buf1010 = reinterpret_tensor(buf463, (8, 196, 1), (196, 1, 1), 0); del buf463  # reuse
    buf1011 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf1012 = reinterpret_tensor(buf448, (8, 49, 1), (49, 1, 1), 0); del buf448  # reuse
    buf1013 = reinterpret_tensor(buf439, (8, 196, 1), (196, 1, 1), 0); del buf439  # reuse
    buf1014 = reinterpret_tensor(buf431, (8, 196, 1), (196, 1, 1), 0); del buf431  # reuse
    buf1015 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf1016 = reinterpret_tensor(buf415, (8, 49, 1), (49, 1, 1), 0); del buf415  # reuse
    buf1017 = reinterpret_tensor(buf406, (8, 196, 1), (196, 1, 1), 0); del buf406  # reuse
    buf1018 = reinterpret_tensor(buf398, (8, 196, 1), (196, 1, 1), 0); del buf398  # reuse
    buf1019 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf1020 = reinterpret_tensor(buf383, (8, 49, 1), (49, 1, 1), 0); del buf383  # reuse
    buf1021 = reinterpret_tensor(buf374, (8, 196, 1), (196, 1, 1), 0); del buf374  # reuse
    buf1022 = reinterpret_tensor(buf366, (8, 196, 1), (196, 1, 1), 0); del buf366  # reuse
    buf1023 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf1024 = reinterpret_tensor(buf350, (8, 49, 1), (49, 1, 1), 0); del buf350  # reuse
    buf1025 = reinterpret_tensor(buf341, (8, 196, 1), (196, 1, 1), 0); del buf341  # reuse
    buf1026 = reinterpret_tensor(buf333, (8, 196, 1), (196, 1, 1), 0); del buf333  # reuse
    buf1027 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf1028 = reinterpret_tensor(buf318, (8, 49, 1), (49, 1, 1), 0); del buf318  # reuse
    buf1029 = reinterpret_tensor(buf309, (8, 196, 1), (196, 1, 1), 0); del buf309  # reuse
    buf1030 = reinterpret_tensor(buf299, (8, 196, 1), (196, 1, 1), 0); del buf299  # reuse
    buf1031 = empty_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cpu', dtype=torch.float32)
    buf1032 = reinterpret_tensor(buf284, (8, 49, 1), (49, 1, 1), 0); del buf284  # reuse
    buf1033 = reinterpret_tensor(buf275, (8, 196, 1), (196, 1, 1), 0); del buf275  # reuse
    buf1034 = reinterpret_tensor(buf271, (8, 196, 1), (196, 1, 1), 0); del buf271  # reuse
    buf1035 = reinterpret_tensor(buf261, (8, 784, 1), (784, 1, 1), 0); del buf261  # reuse
    buf1036 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cpu', dtype=torch.float32)
    buf1037 = reinterpret_tensor(buf246, (8, 49, 1), (49, 1, 1), 0); del buf246  # reuse
    buf1038 = reinterpret_tensor(buf237, (8, 784, 1), (784, 1, 1), 0); del buf237  # reuse
    buf1039 = reinterpret_tensor(buf229, (8, 784, 1), (784, 1, 1), 0); del buf229  # reuse
    buf1040 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cpu', dtype=torch.float32)
    buf1041 = reinterpret_tensor(buf213, (8, 49, 1), (49, 1, 1), 0); del buf213  # reuse
    buf1042 = reinterpret_tensor(buf204, (8, 784, 1), (784, 1, 1), 0); del buf204  # reuse
    buf1043 = reinterpret_tensor(buf196, (8, 784, 1), (784, 1, 1), 0); del buf196  # reuse
    buf1044 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cpu', dtype=torch.float32)
    buf1045 = reinterpret_tensor(buf181, (8, 49, 1), (49, 1, 1), 0); del buf181  # reuse
    buf1046 = reinterpret_tensor(buf172, (8, 784, 1), (784, 1, 1), 0); del buf172  # reuse
    buf1047 = reinterpret_tensor(buf162, (8, 784, 1), (784, 1, 1), 0); del buf162  # reuse
    buf1048 = empty_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cpu', dtype=torch.float32)
    buf1049 = reinterpret_tensor(buf147, (8, 49, 1), (49, 1, 1), 0); del buf147  # reuse
    buf1050 = reinterpret_tensor(buf138, (8, 784, 1), (784, 1, 1), 0); del buf138  # reuse
    buf1051 = reinterpret_tensor(buf134, (8, 784, 1), (784, 1, 1), 0); del buf134  # reuse
    buf1052 = reinterpret_tensor(buf124, (8, 3136, 1), (3136, 1, 1), 0); del buf124  # reuse
    buf1053 = reinterpret_tensor(buf108, (8, 49, 1), (49, 1, 1), 0); del buf108  # reuse
    buf1054 = reinterpret_tensor(buf100, (8, 3136, 1), (3136, 1, 1), 0); del buf100  # reuse
    buf1055 = reinterpret_tensor(buf92, (8, 3136, 1), (3136, 1, 1), 0); del buf92  # reuse
    buf1056 = reinterpret_tensor(buf77, (8, 49, 1), (49, 1, 1), 0); del buf77  # reuse
    buf1057 = reinterpret_tensor(buf69, (8, 3136, 1), (3136, 1, 1), 0); del buf69  # reuse
    buf1058 = reinterpret_tensor(buf59, (8, 3136, 1), (3136, 1, 1), 0); del buf59  # reuse
    buf1059 = reinterpret_tensor(buf44, (8, 49, 1), (49, 1, 1), 0); del buf44  # reuse
    buf1060 = reinterpret_tensor(buf36, (8, 3136, 1), (3136, 1, 1), 0); del buf36  # reuse
    buf1061 = reinterpret_tensor(buf32, (8, 3136, 1), (3136, 1, 1), 0); del buf32  # reuse
    cpp_fused_add_detach_native_layer_norm_native_layer_norm_backward_143(c_void_p(buf951.data_ptr()), c_void_p(buf952.data_ptr()), c_void_p(buf954.data_ptr()), c_void_p(buf955.data_ptr()), c_void_p(buf957.data_ptr()), c_void_p(buf958.data_ptr()), c_void_p(buf960.data_ptr()), c_void_p(buf961.data_ptr()), c_void_p(buf962.data_ptr()), c_void_p(buf964.data_ptr()), c_void_p(buf965.data_ptr()), c_void_p(buf966.data_ptr()), c_void_p(buf968.data_ptr()), c_void_p(buf969.data_ptr()), c_void_p(buf970.data_ptr()), c_void_p(buf972.data_ptr()), c_void_p(buf973.data_ptr()), c_void_p(buf974.data_ptr()), c_void_p(buf976.data_ptr()), c_void_p(buf977.data_ptr()), c_void_p(buf978.data_ptr()), c_void_p(buf980.data_ptr()), c_void_p(buf981.data_ptr()), c_void_p(buf982.data_ptr()), c_void_p(buf984.data_ptr()), c_void_p(buf985.data_ptr()), c_void_p(buf986.data_ptr()), c_void_p(buf988.data_ptr()), c_void_p(buf989.data_ptr()), c_void_p(buf990.data_ptr()), c_void_p(buf992.data_ptr()), c_void_p(buf993.data_ptr()), c_void_p(buf994.data_ptr()), c_void_p(buf996.data_ptr()), c_void_p(buf997.data_ptr()), c_void_p(buf998.data_ptr()), c_void_p(buf1000.data_ptr()), c_void_p(buf1001.data_ptr()), c_void_p(buf1002.data_ptr()), c_void_p(buf1004.data_ptr()), c_void_p(buf1005.data_ptr()), c_void_p(buf1006.data_ptr()), c_void_p(buf1008.data_ptr()), c_void_p(buf1009.data_ptr()), c_void_p(buf1010.data_ptr()), c_void_p(buf1012.data_ptr()), c_void_p(buf1013.data_ptr()), c_void_p(buf1014.data_ptr()), c_void_p(buf1016.data_ptr()), c_void_p(buf1017.data_ptr()), c_void_p(buf1018.data_ptr()), c_void_p(buf1020.data_ptr()), c_void_p(buf1021.data_ptr()), c_void_p(buf1022.data_ptr()), c_void_p(buf1024.data_ptr()), c_void_p(buf1025.data_ptr()), c_void_p(buf1026.data_ptr()), c_void_p(buf1028.data_ptr()), c_void_p(buf1029.data_ptr()), c_void_p(buf1030.data_ptr()), c_void_p(buf1032.data_ptr()), c_void_p(buf1033.data_ptr()), c_void_p(buf1034.data_ptr()), c_void_p(buf1035.data_ptr()), c_void_p(buf1037.data_ptr()), c_void_p(buf1038.data_ptr()), c_void_p(buf1039.data_ptr()), c_void_p(buf1041.data_ptr()), c_void_p(buf1042.data_ptr()), c_void_p(buf1043.data_ptr()), c_void_p(buf1045.data_ptr()), c_void_p(buf1046.data_ptr()), c_void_p(buf1047.data_ptr()), c_void_p(buf1049.data_ptr()), c_void_p(buf1050.data_ptr()), c_void_p(buf1051.data_ptr()), c_void_p(buf1052.data_ptr()), c_void_p(buf1053.data_ptr()), c_void_p(buf1054.data_ptr()), c_void_p(buf1055.data_ptr()), c_void_p(buf1056.data_ptr()), c_void_p(buf1057.data_ptr()), c_void_p(buf1058.data_ptr()), c_void_p(buf1059.data_ptr()), c_void_p(buf1060.data_ptr()), c_void_p(buf1061.data_ptr()), c_void_p(buf927.data_ptr()), c_void_p(buf902.data_ptr()), c_void_p(buf875.data_ptr()), c_void_p(buf844.data_ptr()), c_void_p(buf811.data_ptr()), c_void_p(buf779.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf953.data_ptr()), c_void_p(buf956.data_ptr()), c_void_p(buf959.data_ptr()), c_void_p(buf963.data_ptr()), c_void_p(buf967.data_ptr()), c_void_p(buf971.data_ptr()), c_void_p(buf975.data_ptr()), c_void_p(buf979.data_ptr()), c_void_p(buf983.data_ptr()), c_void_p(buf987.data_ptr()), c_void_p(buf991.data_ptr()), c_void_p(buf995.data_ptr()), c_void_p(buf999.data_ptr()), c_void_p(buf1003.data_ptr()), c_void_p(buf1007.data_ptr()), c_void_p(buf1011.data_ptr()), c_void_p(buf1015.data_ptr()), c_void_p(buf1019.data_ptr()), c_void_p(buf1023.data_ptr()), c_void_p(buf1027.data_ptr()), c_void_p(buf1031.data_ptr()), c_void_p(buf1036.data_ptr()), c_void_p(buf1040.data_ptr()), c_void_p(buf1044.data_ptr()), c_void_p(buf1048.data_ptr()))
    return (buf950, buf0, primals_3, primals_5, buf1, primals_11, primals_17, primals_23, primals_25, buf2, primals_31, primals_37, primals_43, buf3, primals_49, primals_55, buf4, primals_63, primals_65, buf5, primals_71, primals_77, primals_83, primals_85, buf6, primals_91, primals_97, primals_103, buf7, primals_109, primals_115, primals_121, buf8, primals_127, primals_133, buf9, primals_141, primals_143, buf10, primals_149, primals_155, primals_161, primals_163, buf11, primals_169, primals_175, primals_181, buf12, primals_187, primals_193, primals_199, buf13, primals_205, primals_211, primals_217, buf14, primals_223, primals_229, primals_235, buf15, primals_241, primals_247, primals_253, buf16, primals_259, primals_265, primals_271, buf17, primals_277, primals_283, primals_289, buf18, primals_295, primals_301, primals_307, buf19, primals_313, primals_319, primals_325, buf20, primals_331, primals_337, primals_343, buf21, primals_349, primals_355, primals_361, buf22, primals_367, primals_373, primals_379, buf23, primals_385, primals_391, primals_397, buf24, primals_403, primals_409, primals_415, buf25, primals_421, primals_427, primals_433, buf26, primals_439, primals_445, primals_451, buf27, primals_457, primals_463, buf28, primals_471, primals_473, primals_481, primals_487, primals_489, primals_497, primals_503, primals_511, primals_517, buf29, buf34, buf38, buf39, reinterpret_tensor(buf40, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), buf41, buf46, buf47, reinterpret_tensor(buf48, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf48, (8, 1, 49, 64), (6272, 0, 128, 1), 64), buf51, buf52, buf53, buf54, buf55, reinterpret_tensor(buf50, (25088, 64), (64, 1), 0), buf61, buf62, buf63, buf64, buf66, buf71, buf72, reinterpret_tensor(buf73, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), buf74, buf79, buf80, reinterpret_tensor(buf81, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf81, (8, 1, 49, 64), (6272, 0, 128, 1), 64), buf84, buf85, buf86, buf87, buf88, reinterpret_tensor(buf83, (25088, 64), (64, 1), 0), buf94, buf95, buf96, buf97, buf102, buf103, reinterpret_tensor(buf104, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), buf105, buf110, buf111, reinterpret_tensor(buf112, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf112, (8, 1, 49, 64), (6272, 0, 128, 1), 64), buf115, buf116, buf117, buf118, buf119, reinterpret_tensor(buf114, (25088, 64), (64, 1), 0), buf126, buf127, buf128, buf129, buf131, buf136, buf140, buf141, buf143, buf144, buf149, buf150, reinterpret_tensor(buf151, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf151, (8, 2, 49, 64), (12544, 64, 256, 1), 128), buf154, buf155, buf156, buf157, buf158, reinterpret_tensor(buf153, (6272, 128), (128, 1), 0), buf164, buf165, buf166, buf167, buf169, buf174, buf175, buf177, buf178, buf183, buf184, reinterpret_tensor(buf185, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf185, (8, 2, 49, 64), (12544, 64, 256, 1), 128), buf188, buf189, buf190, buf191, buf192, reinterpret_tensor(buf187, (6272, 128), (128, 1), 0), buf198, buf199, buf200, buf201, buf206, buf207, buf209, buf210, buf215, buf216, reinterpret_tensor(buf217, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf217, (8, 2, 49, 64), (12544, 64, 256, 1), 128), buf220, buf221, buf222, buf223, buf224, reinterpret_tensor(buf219, (6272, 128), (128, 1), 0), buf231, buf232, buf233, buf234, buf239, buf240, buf242, buf243, buf248, buf249, reinterpret_tensor(buf250, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf250, (8, 2, 49, 64), (12544, 64, 256, 1), 128), buf253, buf254, buf255, buf256, buf257, reinterpret_tensor(buf252, (6272, 128), (128, 1), 0), buf263, buf264, buf265, buf266, buf268, buf273, buf277, buf278, buf280, buf281, buf286, buf287, reinterpret_tensor(buf288, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf288, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf291, buf292, buf293, buf294, buf295, reinterpret_tensor(buf290, (1568, 320), (320, 1), 0), buf301, buf302, buf303, buf304, buf306, buf311, buf312, buf314, buf315, buf320, buf321, reinterpret_tensor(buf322, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf322, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf325, buf326, buf327, buf328, buf329, reinterpret_tensor(buf324, (1568, 320), (320, 1), 0), buf335, buf336, buf337, buf338, buf343, buf344, buf346, buf347, buf352, buf353, reinterpret_tensor(buf354, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf354, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf357, buf358, buf359, buf360, buf361, reinterpret_tensor(buf356, (1568, 320), (320, 1), 0), buf368, buf369, buf370, buf371, buf376, buf377, buf379, buf380, buf385, buf386, reinterpret_tensor(buf387, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf387, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf390, buf391, buf392, buf393, buf394, reinterpret_tensor(buf389, (1568, 320), (320, 1), 0), buf400, buf401, buf402, buf403, buf408, buf409, buf411, buf412, buf417, buf418, reinterpret_tensor(buf419, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf419, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf422, buf423, buf424, buf425, buf426, reinterpret_tensor(buf421, (1568, 320), (320, 1), 0), buf433, buf434, buf435, buf436, buf441, buf442, buf444, buf445, buf450, buf451, reinterpret_tensor(buf452, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf452, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf455, buf456, buf457, buf458, buf459, reinterpret_tensor(buf454, (1568, 320), (320, 1), 0), buf465, buf466, buf467, buf468, buf473, buf474, buf476, buf477, buf482, buf483, reinterpret_tensor(buf484, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf484, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf487, buf488, buf489, buf490, buf491, reinterpret_tensor(buf486, (1568, 320), (320, 1), 0), buf498, buf499, buf500, buf501, buf506, buf507, buf509, buf510, buf515, buf516, reinterpret_tensor(buf517, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf517, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf520, buf521, buf522, buf523, buf524, reinterpret_tensor(buf519, (1568, 320), (320, 1), 0), buf530, buf531, buf532, buf533, buf538, buf539, buf541, buf542, buf547, buf548, reinterpret_tensor(buf549, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf549, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf552, buf553, buf554, buf555, buf556, reinterpret_tensor(buf551, (1568, 320), (320, 1), 0), buf563, buf564, buf565, buf566, buf571, buf572, buf574, buf575, buf580, buf581, reinterpret_tensor(buf582, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf582, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf585, buf586, buf587, buf588, buf589, reinterpret_tensor(buf584, (1568, 320), (320, 1), 0), buf595, buf596, buf597, buf598, buf603, buf604, buf606, buf607, buf612, buf613, reinterpret_tensor(buf614, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf614, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf617, buf618, buf619, buf620, buf621, reinterpret_tensor(buf616, (1568, 320), (320, 1), 0), buf628, buf629, buf630, buf631, buf636, buf637, buf639, buf640, buf645, buf646, reinterpret_tensor(buf647, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf647, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf650, buf651, buf652, buf653, buf654, reinterpret_tensor(buf649, (1568, 320), (320, 1), 0), buf660, buf661, buf662, buf663, buf668, buf669, buf671, buf672, buf677, buf678, reinterpret_tensor(buf679, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf679, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf682, buf683, buf684, buf685, buf686, reinterpret_tensor(buf681, (1568, 320), (320, 1), 0), buf693, buf694, buf695, buf696, buf701, buf702, buf704, buf705, buf710, buf711, reinterpret_tensor(buf712, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf712, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf715, buf716, buf717, buf718, buf719, reinterpret_tensor(buf714, (1568, 320), (320, 1), 0), buf725, buf726, buf727, buf728, buf733, buf734, buf736, buf737, buf742, buf743, reinterpret_tensor(buf744, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf744, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf747, buf748, buf749, buf750, buf751, reinterpret_tensor(buf746, (1568, 320), (320, 1), 0), buf758, buf759, buf760, buf761, buf766, buf767, buf769, buf770, buf775, buf776, reinterpret_tensor(buf777, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf777, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf780, buf781, buf782, buf783, buf784, reinterpret_tensor(buf779, (1568, 320), (320, 1), 0), buf790, buf791, buf792, buf793, buf798, buf799, buf801, buf802, buf807, buf808, reinterpret_tensor(buf809, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf809, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf812, buf813, buf814, buf815, buf816, reinterpret_tensor(buf811, (1568, 320), (320, 1), 0), buf823, buf824, buf825, buf826, buf831, buf832, buf834, buf835, buf840, buf841, reinterpret_tensor(buf842, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf842, (8, 5, 49, 64), (31360, 64, 640, 1), 320), buf845, buf846, buf847, buf848, buf849, reinterpret_tensor(buf844, (1568, 320), (320, 1), 0), buf855, buf856, buf857, buf858, buf860, buf865, buf869, buf870, buf872, reinterpret_tensor(buf873, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf873, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), buf876, buf877, buf878, buf879, buf880, reinterpret_tensor(buf875, (392, 512), (512, 1), 0), buf886, buf887, buf888, buf889, buf891, buf896, buf897, buf899, reinterpret_tensor(buf900, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf900, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), buf903, buf904, buf905, buf906, buf907, reinterpret_tensor(buf902, (392, 512), (512, 1), 0), buf913, buf914, buf915, buf916, buf921, buf922, buf924, reinterpret_tensor(buf925, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf925, (8, 8, 49, 64), (50176, 64, 1024, 1), 512), buf928, buf929, buf930, buf931, buf932, reinterpret_tensor(buf927, (392, 512), (512, 1), 0), buf939, buf940, buf941, buf942, buf947, buf949, reinterpret_tensor(primals_519, (1000, 512), (512, 1), 0), buf951, reinterpret_tensor(primals_515, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_513, (2048, 512), (512, 1), 0), buf952, reinterpret_tensor(primals_509, (512, 512), (512, 1), 0), buf953, reinterpret_tensor(primals_507, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_505, (512, 512), (512, 1), 0), buf954, reinterpret_tensor(primals_501, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_499, (2048, 512), (512, 1), 0), buf955, reinterpret_tensor(primals_495, (512, 512), (512, 1), 0), buf956, reinterpret_tensor(primals_493, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_491, (512, 512), (512, 1), 0), buf957, reinterpret_tensor(primals_485, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_483, (2048, 512), (512, 1), 0), buf958, reinterpret_tensor(primals_479, (512, 512), (512, 1), 0), buf959, reinterpret_tensor(primals_477, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_475, (512, 512), (512, 1), 0), buf960, buf961, reinterpret_tensor(primals_467, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_465, (1280, 320), (320, 1), 0), buf962, reinterpret_tensor(primals_461, (320, 320), (320, 1), 0), buf963, reinterpret_tensor(primals_459, (640, 320), (320, 1), 0), buf964, reinterpret_tensor(primals_453, (320, 320), (320, 1), 0), buf965, reinterpret_tensor(primals_449, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_447, (1280, 320), (320, 1), 0), buf966, reinterpret_tensor(primals_443, (320, 320), (320, 1), 0), buf967, reinterpret_tensor(primals_441, (640, 320), (320, 1), 0), buf968, reinterpret_tensor(primals_435, (320, 320), (320, 1), 0), buf969, reinterpret_tensor(primals_431, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_429, (1280, 320), (320, 1), 0), buf970, reinterpret_tensor(primals_425, (320, 320), (320, 1), 0), buf971, reinterpret_tensor(primals_423, (640, 320), (320, 1), 0), buf972, reinterpret_tensor(primals_417, (320, 320), (320, 1), 0), buf973, reinterpret_tensor(primals_413, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_411, (1280, 320), (320, 1), 0), buf974, reinterpret_tensor(primals_407, (320, 320), (320, 1), 0), buf975, reinterpret_tensor(primals_405, (640, 320), (320, 1), 0), buf976, reinterpret_tensor(primals_399, (320, 320), (320, 1), 0), buf977, reinterpret_tensor(primals_395, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_393, (1280, 320), (320, 1), 0), buf978, reinterpret_tensor(primals_389, (320, 320), (320, 1), 0), buf979, reinterpret_tensor(primals_387, (640, 320), (320, 1), 0), buf980, reinterpret_tensor(primals_381, (320, 320), (320, 1), 0), buf981, reinterpret_tensor(primals_377, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_375, (1280, 320), (320, 1), 0), buf982, reinterpret_tensor(primals_371, (320, 320), (320, 1), 0), buf983, reinterpret_tensor(primals_369, (640, 320), (320, 1), 0), buf984, reinterpret_tensor(primals_363, (320, 320), (320, 1), 0), buf985, reinterpret_tensor(primals_359, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_357, (1280, 320), (320, 1), 0), buf986, reinterpret_tensor(primals_353, (320, 320), (320, 1), 0), buf987, reinterpret_tensor(primals_351, (640, 320), (320, 1), 0), buf988, reinterpret_tensor(primals_345, (320, 320), (320, 1), 0), buf989, reinterpret_tensor(primals_341, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_339, (1280, 320), (320, 1), 0), buf990, reinterpret_tensor(primals_335, (320, 320), (320, 1), 0), buf991, reinterpret_tensor(primals_333, (640, 320), (320, 1), 0), buf992, reinterpret_tensor(primals_327, (320, 320), (320, 1), 0), buf993, reinterpret_tensor(primals_323, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_321, (1280, 320), (320, 1), 0), buf994, reinterpret_tensor(primals_317, (320, 320), (320, 1), 0), buf995, reinterpret_tensor(primals_315, (640, 320), (320, 1), 0), buf996, reinterpret_tensor(primals_309, (320, 320), (320, 1), 0), buf997, reinterpret_tensor(primals_305, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_303, (1280, 320), (320, 1), 0), buf998, reinterpret_tensor(primals_299, (320, 320), (320, 1), 0), buf999, reinterpret_tensor(primals_297, (640, 320), (320, 1), 0), buf1000, reinterpret_tensor(primals_291, (320, 320), (320, 1), 0), buf1001, reinterpret_tensor(primals_287, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_285, (1280, 320), (320, 1), 0), buf1002, reinterpret_tensor(primals_281, (320, 320), (320, 1), 0), buf1003, reinterpret_tensor(primals_279, (640, 320), (320, 1), 0), buf1004, reinterpret_tensor(primals_273, (320, 320), (320, 1), 0), buf1005, reinterpret_tensor(primals_269, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_267, (1280, 320), (320, 1), 0), buf1006, reinterpret_tensor(primals_263, (320, 320), (320, 1), 0), buf1007, reinterpret_tensor(primals_261, (640, 320), (320, 1), 0), buf1008, reinterpret_tensor(primals_255, (320, 320), (320, 1), 0), buf1009, reinterpret_tensor(primals_251, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_249, (1280, 320), (320, 1), 0), buf1010, reinterpret_tensor(primals_245, (320, 320), (320, 1), 0), buf1011, reinterpret_tensor(primals_243, (640, 320), (320, 1), 0), buf1012, reinterpret_tensor(primals_237, (320, 320), (320, 1), 0), buf1013, reinterpret_tensor(primals_233, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_231, (1280, 320), (320, 1), 0), buf1014, reinterpret_tensor(primals_227, (320, 320), (320, 1), 0), buf1015, reinterpret_tensor(primals_225, (640, 320), (320, 1), 0), buf1016, reinterpret_tensor(primals_219, (320, 320), (320, 1), 0), buf1017, reinterpret_tensor(primals_215, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_213, (1280, 320), (320, 1), 0), buf1018, reinterpret_tensor(primals_209, (320, 320), (320, 1), 0), buf1019, reinterpret_tensor(primals_207, (640, 320), (320, 1), 0), buf1020, reinterpret_tensor(primals_201, (320, 320), (320, 1), 0), buf1021, reinterpret_tensor(primals_197, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_195, (1280, 320), (320, 1), 0), buf1022, reinterpret_tensor(primals_191, (320, 320), (320, 1), 0), buf1023, reinterpret_tensor(primals_189, (640, 320), (320, 1), 0), buf1024, reinterpret_tensor(primals_183, (320, 320), (320, 1), 0), buf1025, reinterpret_tensor(primals_179, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_177, (1280, 320), (320, 1), 0), buf1026, reinterpret_tensor(primals_173, (320, 320), (320, 1), 0), buf1027, reinterpret_tensor(primals_171, (640, 320), (320, 1), 0), buf1028, reinterpret_tensor(primals_165, (320, 320), (320, 1), 0), buf1029, reinterpret_tensor(primals_159, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_157, (1280, 320), (320, 1), 0), buf1030, reinterpret_tensor(primals_153, (320, 320), (320, 1), 0), buf1031, reinterpret_tensor(primals_151, (640, 320), (320, 1), 0), buf1032, reinterpret_tensor(primals_145, (320, 320), (320, 1), 0), buf1033, buf1034, reinterpret_tensor(primals_137, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_135, (1024, 128), (128, 1), 0), buf1035, reinterpret_tensor(primals_131, (128, 128), (128, 1), 0), buf1036, reinterpret_tensor(primals_129, (256, 128), (128, 1), 0), buf1037, reinterpret_tensor(primals_123, (128, 128), (128, 1), 0), buf1038, reinterpret_tensor(primals_119, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_117, (1024, 128), (128, 1), 0), buf1039, reinterpret_tensor(primals_113, (128, 128), (128, 1), 0), buf1040, reinterpret_tensor(primals_111, (256, 128), (128, 1), 0), buf1041, reinterpret_tensor(primals_105, (128, 128), (128, 1), 0), buf1042, reinterpret_tensor(primals_101, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_99, (1024, 128), (128, 1), 0), buf1043, reinterpret_tensor(primals_95, (128, 128), (128, 1), 0), buf1044, reinterpret_tensor(primals_93, (256, 128), (128, 1), 0), buf1045, reinterpret_tensor(primals_87, (128, 128), (128, 1), 0), buf1046, reinterpret_tensor(primals_81, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_79, (1024, 128), (128, 1), 0), buf1047, reinterpret_tensor(primals_75, (128, 128), (128, 1), 0), buf1048, reinterpret_tensor(primals_73, (256, 128), (128, 1), 0), buf1049, reinterpret_tensor(primals_67, (128, 128), (128, 1), 0), buf1050, buf1051, reinterpret_tensor(primals_59, (64, 512), (512, 1), 0), reinterpret_tensor(primals_57, (512, 64), (64, 1), 0), buf1052, reinterpret_tensor(primals_53, (64, 64), (64, 1), 0), buf114, reinterpret_tensor(primals_51, (128, 64), (64, 1), 0), buf1053, reinterpret_tensor(primals_45, (64, 64), (64, 1), 0), buf1054, reinterpret_tensor(primals_41, (64, 512), (512, 1), 0), reinterpret_tensor(primals_39, (512, 64), (64, 1), 0), buf1055, reinterpret_tensor(primals_35, (64, 64), (64, 1), 0), buf83, reinterpret_tensor(primals_33, (128, 64), (64, 1), 0), buf1056, reinterpret_tensor(primals_27, (64, 64), (64, 1), 0), buf1057, reinterpret_tensor(primals_21, (64, 512), (512, 1), 0), reinterpret_tensor(primals_19, (512, 64), (64, 1), 0), buf1058, reinterpret_tensor(primals_15, (64, 64), (64, 1), 0), buf50, reinterpret_tensor(primals_13, (128, 64), (64, 1), 0), buf1059, reinterpret_tensor(primals_7, (64, 64), (64, 1), 0), buf1060, buf1061, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((128, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((128, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((320, 128, 2, 2), (512, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_346 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_348 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_349 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_351 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_353 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_354 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_355 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_357 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_358 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_359 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_360 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_361 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_362 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_363 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_364 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_365 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_366 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_367 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_368 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_369 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_370 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_371 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_372 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_373 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_374 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_375 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_376 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_377 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_378 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_379 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_380 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_381 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_382 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_383 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_384 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_385 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_386 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_387 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_388 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_389 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_390 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_391 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_392 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_393 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_394 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_395 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_396 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_397 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_398 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_399 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_400 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_401 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_402 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_403 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_404 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_405 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_406 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_407 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_408 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_409 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_410 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_411 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_412 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_413 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_414 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_415 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_416 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_417 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_418 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_419 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_420 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_421 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_422 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_423 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_424 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_425 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_426 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_427 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_428 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_429 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_430 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_431 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_432 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_433 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_434 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_435 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_436 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_437 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_438 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_439 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_440 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_441 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_442 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_443 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_444 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_445 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_446 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_447 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_448 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_449 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_450 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_451 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_452 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_453 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_454 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_455 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_456 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_457 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_458 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_459 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_460 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_461 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_462 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_463 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_464 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_465 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_466 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_467 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_468 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_469 = rand_strided((512, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_470 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_471 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_472 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_473 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_474 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_475 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_476 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_477 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_478 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_479 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_480 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_481 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_482 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_483 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_484 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_485 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_486 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_487 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_488 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_489 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_490 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_491 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_492 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_493 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_494 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_495 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_496 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_497 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_498 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_499 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_500 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_501 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_502 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_503 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_504 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_505 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_506 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_507 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_508 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_509 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_510 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_511 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_512 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_513 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_514 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_515 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_516 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_517 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_518 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_519 = rand_strided((1000, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_520 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_521 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('twins_pcpvt_base', benchmark_compiled_module)
