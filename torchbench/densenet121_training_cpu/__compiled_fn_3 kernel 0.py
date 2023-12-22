
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
                       const float* in_ptr34,
                       const float* in_ptr35,
                       const float* in_ptr36,
                       const float* in_ptr37,
                       const float* in_ptr38,
                       const float* in_ptr39,
                       const float* in_ptr40,
                       const float* in_ptr41,
                       const float* in_ptr42,
                       const float* in_ptr43,
                       const float* in_ptr44,
                       const float* in_ptr45,
                       const float* in_ptr46,
                       const float* in_ptr47,
                       const float* in_ptr48,
                       const float* in_ptr49,
                       const float* in_ptr50,
                       const float* in_ptr51,
                       const float* in_ptr52,
                       const float* in_ptr53,
                       const float* in_ptr54,
                       const float* in_ptr55,
                       const float* in_ptr56,
                       const float* in_ptr57,
                       const float* in_ptr58,
                       const float* in_ptr59,
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
                       float* out_ptr33,
                       float* out_ptr34,
                       float* out_ptr35,
                       float* out_ptr36,
                       float* out_ptr37,
                       float* out_ptr38,
                       float* out_ptr39,
                       float* out_ptr40,
                       float* out_ptr41,
                       float* out_ptr42,
                       float* out_ptr43,
                       float* out_ptr44,
                       float* out_ptr45,
                       float* out_ptr46,
                       float* out_ptr47,
                       float* out_ptr48,
                       float* out_ptr49,
                       float* out_ptr50,
                       float* out_ptr51,
                       float* out_ptr52,
                       float* out_ptr53,
                       float* out_ptr54,
                       float* out_ptr55,
                       float* out_ptr56,
                       float* out_ptr57,
                       float* out_ptr58,
                       float* out_ptr59)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (147L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (147L*x0))] = tmp0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr8[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr8 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr9 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr9[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr9 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr10 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr10[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr10 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr11 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr11[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr11 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr12 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr12[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr12 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr13 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr13[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr13 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr14 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr14[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr14 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr17 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr17[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr17 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr18 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr18[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr18 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr19 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr19[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr19 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr20 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr20[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr20 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr21 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr21[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr21 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr22 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr22[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr22 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr23 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr23[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr23 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr24 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr24[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr24 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr25 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr25[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr25 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr26 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr26[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr26 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr27 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr27[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr27 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr28 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr28 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr28[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr28 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr29 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr29[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr29 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr30 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr30 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr30[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr30 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr31 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr31 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr31[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr31 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr32 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr32 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr32[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr32 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr33 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr33 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr33[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr33 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr34 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr34 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr34[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr34 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr35 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr35 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr35[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr35 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr36 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr36 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr36[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr36 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr37 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr37 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr37[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr37 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr38 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr38 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr38[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr38 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr39 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr39 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr39[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr39 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr40 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr40 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr40[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr40 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr41 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr41 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr41[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr41 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr42 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr42 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr42[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr42 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr43 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr43 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr43[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr43 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr44 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr44 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr44[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr44 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr45 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr45 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr45[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr45 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr46 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr46 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr46[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr46 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr47 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr47 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr47[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr47 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr48 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr48 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr48[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr48 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr49 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr49 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr49[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr49 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr50 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr50 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr50[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr50 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr51 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr51 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr51[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr51 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr52 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr52 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr52[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr52 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr53 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr53 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr53[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr53 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr54 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr54 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr54[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr54 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr55 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr55 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr55[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr55 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr56 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr56 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr56[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr56 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr57 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr57 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr57[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr57 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr58 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr58 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr58[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr58 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                }
            }
        }
    }
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
                        auto tmp0 = in_ptr59[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr59[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_1 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(112);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-7232L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : -std::numeric_limits<decltype(tmp11())>::infinity();
                            auto tmp14 = c10::convert<long>(2L*x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = out_ptr0[static_cast<long>((-7168L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : -std::numeric_limits<decltype(tmp19())>::infinity();
                            auto tmp22 = max_propagate_nan(tmp21, tmp13);
                            auto tmp23 = c10::convert<long>(1L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = out_ptr0[static_cast<long>((-7104L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : -std::numeric_limits<decltype(tmp28())>::infinity();
                            auto tmp31 = max_propagate_nan(tmp30, tmp22);
                            auto tmp32 = c10::convert<long>(2L*x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = out_ptr0[static_cast<long>((-64L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : -std::numeric_limits<decltype(tmp37())>::infinity();
                            auto tmp40 = max_propagate_nan(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = out_ptr0[static_cast<long>(x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = max_propagate_nan(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = out_ptr0[static_cast<long>(64L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : -std::numeric_limits<decltype(tmp47())>::infinity();
                            auto tmp50 = max_propagate_nan(tmp49, tmp45);
                            auto tmp51 = c10::convert<long>(1L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = out_ptr0[static_cast<long>(7104L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : -std::numeric_limits<decltype(tmp56())>::infinity();
                            auto tmp59 = max_propagate_nan(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = out_ptr0[static_cast<long>(7168L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : -std::numeric_limits<decltype(tmp61())>::infinity();
                            auto tmp64 = max_propagate_nan(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = out_ptr0[static_cast<long>(7232L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : -std::numeric_limits<decltype(tmp66())>::infinity();
                            auto tmp69 = max_propagate_nan(tmp68, tmp64);
                            auto tmp70 = [&]
                            {
                                auto tmp71 = out_ptr0[static_cast<long>((-7232L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp10 ? tmp70() : -std::numeric_limits<decltype(tmp70())>::infinity();
                            auto tmp73 = [&]
                            {
                                auto tmp74 = out_ptr0[static_cast<long>((-7168L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp74;
                            }
                            ;
                            auto tmp75 = tmp18 ? tmp73() : -std::numeric_limits<decltype(tmp73())>::infinity();
                            auto tmp76 = tmp75 > tmp72;
                            auto tmp77 = c10::convert<long>((-112L) + (2L*x2) + (224L*x1));
                            auto tmp78 = c10::convert<long>((-113L) + (2L*x2) + (224L*x1));
                            auto tmp79 = tmp76 ? tmp77 : tmp78;
                            auto tmp80 = max_propagate_nan(tmp75, tmp72);
                            auto tmp81 = [&]
                            {
                                auto tmp82 = out_ptr0[static_cast<long>((-7104L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp82;
                            }
                            ;
                            auto tmp83 = tmp27 ? tmp81() : -std::numeric_limits<decltype(tmp81())>::infinity();
                            auto tmp84 = tmp83 > tmp80;
                            auto tmp85 = c10::convert<long>((-111L) + (2L*x2) + (224L*x1));
                            auto tmp86 = tmp84 ? tmp85 : tmp79;
                            auto tmp87 = max_propagate_nan(tmp83, tmp80);
                            auto tmp88 = [&]
                            {
                                auto tmp89 = out_ptr0[static_cast<long>((-64L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp89;
                            }
                            ;
                            auto tmp90 = tmp36 ? tmp88() : -std::numeric_limits<decltype(tmp88())>::infinity();
                            auto tmp91 = tmp90 > tmp87;
                            auto tmp92 = c10::convert<long>((-1L) + (2L*x2) + (224L*x1));
                            auto tmp93 = tmp91 ? tmp92 : tmp86;
                            auto tmp94 = max_propagate_nan(tmp90, tmp87);
                            auto tmp95 = [&]
                            {
                                auto tmp96 = out_ptr0[static_cast<long>(x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp96;
                            }
                            ;
                            auto tmp97 = tmp41 ? tmp95() : -std::numeric_limits<decltype(tmp95())>::infinity();
                            auto tmp98 = tmp97 > tmp94;
                            auto tmp99 = c10::convert<long>((2L*x2) + (224L*x1));
                            auto tmp100 = tmp98 ? tmp99 : tmp93;
                            auto tmp101 = max_propagate_nan(tmp97, tmp94);
                            auto tmp102 = [&]
                            {
                                auto tmp103 = out_ptr0[static_cast<long>(64L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp46 ? tmp102() : -std::numeric_limits<decltype(tmp102())>::infinity();
                            auto tmp105 = tmp104 > tmp101;
                            auto tmp106 = c10::convert<long>(1L + (2L*x2) + (224L*x1));
                            auto tmp107 = tmp105 ? tmp106 : tmp100;
                            auto tmp108 = max_propagate_nan(tmp104, tmp101);
                            auto tmp109 = [&]
                            {
                                auto tmp110 = out_ptr0[static_cast<long>(7104L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp110;
                            }
                            ;
                            auto tmp111 = tmp55 ? tmp109() : -std::numeric_limits<decltype(tmp109())>::infinity();
                            auto tmp112 = tmp111 > tmp108;
                            auto tmp113 = c10::convert<long>(111L + (2L*x2) + (224L*x1));
                            auto tmp114 = tmp112 ? tmp113 : tmp107;
                            auto tmp115 = max_propagate_nan(tmp111, tmp108);
                            auto tmp116 = [&]
                            {
                                auto tmp117 = out_ptr0[static_cast<long>(7168L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp117;
                            }
                            ;
                            auto tmp118 = tmp60 ? tmp116() : -std::numeric_limits<decltype(tmp116())>::infinity();
                            auto tmp119 = tmp118 > tmp115;
                            auto tmp120 = c10::convert<long>(112L + (2L*x2) + (224L*x1));
                            auto tmp121 = tmp119 ? tmp120 : tmp114;
                            auto tmp122 = max_propagate_nan(tmp118, tmp115);
                            auto tmp123 = [&]
                            {
                                auto tmp124 = out_ptr0[static_cast<long>(7232L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp124;
                            }
                            ;
                            auto tmp125 = tmp65 ? tmp123() : -std::numeric_limits<decltype(tmp123())>::infinity();
                            auto tmp126 = tmp125 > tmp122;
                            auto tmp127 = c10::convert<long>(113L + (2L*x2) + (224L*x1));
                            auto tmp128 = tmp126 ? tmp127 : tmp121;
                            auto tmp129 = max_propagate_nan(tmp125, tmp122);
                            out_ptr1[static_cast<long>(x3 + (96L*x2) + (5376L*x1) + (301056L*x0))] = tmp69;
                            out_ptr2[static_cast<long>(x3 + (64L*x2) + (3584L*x1) + (200704L*x0))] = tmp128;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (96L*x2) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                        tmp17.store(out_ptr3 + static_cast<long>(x1 + (64L*x2) + (200704L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (192L*x2) + (602112L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (224L*x2) + (702464L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (224L*x0)));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.cpp('''
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    auto tmp23 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp35 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (96L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(96);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-64L) + x1 + (32L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(128);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-96L) + x1 + (32L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                    auto tmp26 = static_cast<float>(1e-05);
                    auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                    auto tmp28 = std::sqrt(tmp27);
                    auto tmp29 = 1 / tmp28;
                    auto tmp30 = static_cast<float>(1.0);
                    auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                    auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                    auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                    auto tmp37 = tmp36 * (tmp36>0);
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp22;
                    out_ptr1[static_cast<long>(x1 + (128L*x0))] = tmp37;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.cpp('''
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (96L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(96);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-64L) + x1 + (32L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(128);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-96L) + x1 + (32L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(160);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-128L) + x1 + (32L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    auto tmp45 = tmp44 * (tmp44>0);
                    out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp30;
                    out_ptr1[static_cast<long>(x1 + (160L*x0))] = tmp45;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.cpp('''
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
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (224L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (224L*x0)));
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr6 + static_cast<long>(x1 + (192L*x0)));
                    tmp0.store(out_ptr7 + static_cast<long>(x1 + (224L*x0)));
                    tmp0.store(out_ptr8 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr9 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.cpp('''
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
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.cpp('''
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
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (14336L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x2 + (256L*x1) + (14336L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7168L + x2 + (256L*x1) + (14336L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7296L + x2 + (256L*x1) + (14336L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (3584L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(128);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(160);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-128L) + x1 + (32L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp29 = tmp28 * (tmp28>0);
                    out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp14;
                    out_ptr1[static_cast<long>(x1 + (160L*x0))] = tmp29;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp35 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(128);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(160);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-128L) + x1 + (32L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(192);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-160L) + x1 + (32L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                    auto tmp26 = static_cast<float>(1e-05);
                    auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                    auto tmp28 = std::sqrt(tmp27);
                    auto tmp29 = 1 / tmp28;
                    auto tmp30 = static_cast<float>(1.0);
                    auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                    auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                    auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                    auto tmp37 = tmp36 * (tmp36>0);
                    out_ptr0[static_cast<long>(x1 + (192L*x0))] = tmp22;
                    out_ptr1[static_cast<long>(x1 + (192L*x0))] = tmp37;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(128);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(160);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-128L) + x1 + (32L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(192);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-160L) + x1 + (32L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(224);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-192L) + x1 + (32L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    auto tmp45 = tmp44 * (tmp44>0);
                    out_ptr0[static_cast<long>(x1 + (224L*x0))] = tmp30;
                    out_ptr1[static_cast<long>(x1 + (224L*x0))] = tmp45;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_22 = async_compile.cpp('''
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
                       float* out_ptr30)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (352L*x0)));
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (416L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (288L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (320L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (352L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (384L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (416L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (256L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (288L*x0)));
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (320L*x0)));
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (352L*x0)));
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (384L*x0)));
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (416L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (256L*x0)));
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (288L*x0)));
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (320L*x0)));
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (352L*x0)));
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (384L*x0)));
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (416L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (256L*x0)));
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (288L*x0)));
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (320L*x0)));
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (352L*x0)));
                        tmp0.store(out_ptr28 + static_cast<long>(x1 + (384L*x0)));
                        tmp0.store(out_ptr29 + static_cast<long>(x1 + (416L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr30 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_24 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (320L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (352L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (416L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_26 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (352L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (448L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_28 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (352L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (480L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (352L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_30 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_32 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(416L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (416L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (416L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_34 = async_compile.cpp('''
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
                       float* out_ptr21)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr21 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_36 = async_compile.cpp('''
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
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_38 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (14336L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x2 + (512L*x1) + (14336L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7168L + x2 + (512L*x1) + (14336L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7424L + x2 + (512L*x1) + (14336L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (3584L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp17.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(256);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(288);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-256L) + x1 + (32L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp29 = tmp28 * (tmp28>0);
                    out_ptr0[static_cast<long>(x1 + (288L*x0))] = tmp14;
                    out_ptr1[static_cast<long>(x1 + (288L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp35 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(256);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(288);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-256L) + x1 + (32L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(320);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-288L) + x1 + (32L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                    auto tmp26 = static_cast<float>(1e-05);
                    auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                    auto tmp28 = std::sqrt(tmp27);
                    auto tmp29 = 1 / tmp28;
                    auto tmp30 = static_cast<float>(1.0);
                    auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                    auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                    auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                    auto tmp37 = tmp36 * (tmp36>0);
                    out_ptr0[static_cast<long>(x1 + (320L*x0))] = tmp22;
                    out_ptr1[static_cast<long>(x1 + (320L*x0))] = tmp37;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(352L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(256);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(288);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-256L) + x1 + (32L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(320);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-288L) + x1 + (32L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(352);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-320L) + x1 + (32L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    auto tmp45 = tmp44 * (tmp44>0);
                    out_ptr0[static_cast<long>(x1 + (352L*x0))] = tmp30;
                    out_ptr1[static_cast<long>(x1 + (352L*x0))] = tmp45;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_47 = async_compile.cpp('''
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
                       float* out_ptr30)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (416L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (448L*x0)));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (544L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (416L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (512L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (544L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (384L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (416L*x0)));
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (512L*x0)));
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (544L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (384L*x0)));
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (416L*x0)));
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (512L*x0)));
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (544L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (384L*x0)));
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (416L*x0)));
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (448L*x0)));
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (480L*x0)));
                        tmp0.store(out_ptr28 + static_cast<long>(x1 + (512L*x0)));
                        tmp0.store(out_ptr29 + static_cast<long>(x1 + (544L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr30 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_49 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (416L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (544L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(416L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (416L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (416L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_51 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (576L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_53 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (576L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (608L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_55 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (576L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (640L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_57 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (544L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (576L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(544L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (544L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (544L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_59 = async_compile.cpp('''
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
                       float* out_ptr28)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (608L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (576L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (608L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (640L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (576L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (608L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (640L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (576L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (608L*x0)));
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (640L*x0)));
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (576L*x0)));
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (608L*x0)));
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (640L*x0)));
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (576L*x0)));
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (608L*x0)));
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (640L*x0)));
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (576L*x0)));
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (608L*x0)));
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (640L*x0)));
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr28 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_61 = async_compile.cpp('''
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
                       float* out_ptr7,
                       float* out_ptr8)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (704L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (608L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (704L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(608L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (608L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr8 + static_cast<long>(x1 + (608L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_63 = async_compile.cpp('''
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
                       float* out_ptr7,
                       float* out_ptr8)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (736L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (736L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr8 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
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
                       float* out_ptr12)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr12 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_67 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (704L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (736L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (704L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (736L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (800L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (704L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (736L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (768L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (800L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (704L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (736L*x0)));
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (768L*x0)));
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (800L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (704L*x0)));
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (736L*x0)));
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (768L*x0)));
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (800L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (704L*x0)));
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (736L*x0)));
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (768L*x0)));
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (800L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (704L*x0)));
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (736L*x0)));
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (768L*x0)));
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (800L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr28 + static_cast<long>(x1 + (704L*x0)));
                        tmp0.store(out_ptr29 + static_cast<long>(x1 + (736L*x0)));
                        tmp0.store(out_ptr30 + static_cast<long>(x1 + (768L*x0)));
                        tmp0.store(out_ptr31 + static_cast<long>(x1 + (800L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(704L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (704L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr32 + static_cast<long>(x1 + (704L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr9 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (832L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (832L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (832L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr9 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_73 = async_compile.cpp('''
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
                       float* out_ptr12)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (864L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (864L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (864L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (864L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr12 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_75 = async_compile.cpp('''
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
                       float* out_ptr33,
                       float* out_ptr34,
                       float* out_ptr35,
                       float* out_ptr36)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (832L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (864L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr28 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr29 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr30 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr31 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr32 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr33 + static_cast<long>(x1 + (832L*x0)));
                        tmp0.store(out_ptr34 + static_cast<long>(x1 + (864L*x0)));
                        tmp0.store(out_ptr35 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1 + (832L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr36 + static_cast<long>(x1 + (832L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_77 = async_compile.cpp('''
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
                       float* out_ptr12)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(864L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (864L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr12 + static_cast<long>(x1 + (864L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_79 = async_compile.cpp('''
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
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (960L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (960L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (960L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (960L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (960L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr15 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_81 = async_compile.cpp('''
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
                       float* out_ptr33,
                       float* out_ptr34,
                       float* out_ptr35,
                       float* out_ptr36,
                       float* out_ptr37,
                       float* out_ptr38,
                       float* out_ptr39)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (928L*x0)));
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (960L*x0)));
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (992L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr14 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr15 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr16 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr17 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr18 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr19 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr20 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr21 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr22 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr23 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr24 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr25 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr26 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr27 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr28 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr29 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr30 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr31 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr32 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr33 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr34 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr35 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr36 + static_cast<long>(x1 + (928L*x0)));
                        tmp0.store(out_ptr37 + static_cast<long>(x1 + (960L*x0)));
                        tmp0.store(out_ptr38 + static_cast<long>(x1 + (992L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(928L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1 + (928L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr39 + static_cast<long>(x1 + (928L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_83 = async_compile.cpp('''
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
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr15 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_85 = async_compile.cpp('''
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
                       float* out_ptr12)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(992L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (992L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr12 + static_cast<long>(x1 + (992L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_87 = async_compile.cpp('''
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
                       float* out_ptr14)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr12 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1 + (32L*x0)));
                        tmp0.store(out_ptr13 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr14 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_88 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7168L + x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7680L + x2 + (1024L*x1) + (14336L*x0)));
                    auto tmp2 = tmp1 + tmp0;
                    auto tmp4 = tmp3 + tmp2;
                    auto tmp6 = tmp5 + tmp4;
                    auto tmp7 = static_cast<float>(0.25);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (3584L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                tmp17.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(544L); x1+=static_cast<long>(1L))
            {
                auto tmp15 = in_ptr2[static_cast<long>(x1)];
                auto tmp17 = in_ptr3[static_cast<long>(x1)];
                auto tmp25 = in_ptr4[static_cast<long>(x1)];
                auto tmp27 = in_ptr5[static_cast<long>(x1)];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(512);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(544);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr1[static_cast<long>((-512L) + x1 + (32L*x0))];
                    return tmp12;
                }
                ;
                auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp14 = tmp4 ? tmp7 : tmp13;
                auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                auto tmp18 = static_cast<float>(1e-05);
                auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                auto tmp20 = std::sqrt(tmp19);
                auto tmp21 = 1 / tmp20;
                auto tmp22 = static_cast<float>(1.0);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                auto tmp29 = tmp28 * (tmp28>0);
                out_ptr0[static_cast<long>(x1 + (544L*x0))] = tmp14;
                out_ptr1[static_cast<long>(x1 + (544L*x0))] = tmp29;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
            {
                auto tmp23 = in_ptr3[static_cast<long>(x1)];
                auto tmp25 = in_ptr4[static_cast<long>(x1)];
                auto tmp33 = in_ptr5[static_cast<long>(x1)];
                auto tmp35 = in_ptr6[static_cast<long>(x1)];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(512);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(544);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = tmp8 & tmp10;
                auto tmp12 = [&]
                {
                    auto tmp13 = in_ptr1[static_cast<long>((-512L) + x1 + (32L*x0))];
                    return tmp13;
                }
                ;
                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                auto tmp15 = tmp0 >= tmp9;
                auto tmp16 = static_cast<long>(576);
                auto tmp17 = tmp0 < tmp16;
                auto tmp18 = [&]
                {
                    auto tmp19 = in_ptr2[static_cast<long>((-544L) + x1 + (32L*x0))];
                    return tmp19;
                }
                ;
                auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                auto tmp21 = tmp11 ? tmp14 : tmp20;
                auto tmp22 = tmp4 ? tmp7 : tmp21;
                auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                auto tmp26 = static_cast<float>(1e-05);
                auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                auto tmp28 = std::sqrt(tmp27);
                auto tmp29 = 1 / tmp28;
                auto tmp30 = static_cast<float>(1.0);
                auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                auto tmp37 = tmp36 * (tmp36>0);
                out_ptr0[static_cast<long>(x1 + (576L*x0))] = tmp22;
                out_ptr1[static_cast<long>(x1 + (576L*x0))] = tmp37;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(608L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(512);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(544);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-512L) + x1 + (32L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(576);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-544L) + x1 + (32L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(608);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-576L) + x1 + (32L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    auto tmp45 = tmp44 * (tmp44>0);
                    out_ptr0[static_cast<long>(x1 + (608L*x0))] = tmp30;
                    out_ptr1[static_cast<long>(x1 + (608L*x0))] = tmp45;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_96 = async_compile.cpp('''
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
                       float* out_ptr30)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr20 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr21 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr22 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr23 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr24 + static_cast<long>(x1 + (640L*x0)));
                tmp0.store(out_ptr25 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr26 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr27 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr28 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr29 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr30 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_98 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (800L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_100 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (704L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (832L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(704L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (704L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (704L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_102 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (864L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_104 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_106 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (896L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_108 = async_compile.cpp('''
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
                       float* out_ptr28)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr20 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr21 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr22 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr23 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr24 + static_cast<long>(x1 + (832L*x0)));
                tmp0.store(out_ptr25 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr26 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr27 + static_cast<long>(x1 + (928L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (832L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr28 + static_cast<long>(x1 + (832L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_110 = async_compile.cpp('''
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
                       float* out_ptr7,
                       float* out_ptr8)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (864L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (960L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(864L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (864L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr8 + static_cast<long>(x1 + (864L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_112 = async_compile.cpp('''
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
                       float* out_ptr7,
                       float* out_ptr8)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (992L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (896L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (992L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr8 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
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
                       float* out_ptr12)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (928L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(928L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (928L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr12 + static_cast<long>(x1 + (928L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_116 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr6 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr7 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr8 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr9 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr10 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr11 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr12 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr13 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr14 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr15 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr16 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr17 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr18 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr19 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr20 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr21 + static_cast<long>(x1 + (960L*x0)));
                tmp0.store(out_ptr22 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr23 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr24 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr4 + static_cast<long>(x1 + (992L*x0)));
                tmp0.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(992L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (992L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr6 + static_cast<long>(x1 + (992L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_relu_view_120 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                tmp0.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
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


cpp_fused_native_batch_norm_backward_threshold_backward_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       bool* out_ptr0,
                       float* out_ptr1)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x2) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (200704L*x0)));
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_10, (96, ), (1, ))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_12, (128, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_22, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_24, (128, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_28, (192, ), (1, ))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_30, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_34, (224, ), (1, ))
    assert_size_stride(primals_35, (224, ), (1, ))
    assert_size_stride(primals_36, (128, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_42, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_49, (160, ), (1, ))
    assert_size_stride(primals_50, (160, ), (1, ))
    assert_size_stride(primals_51, (128, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_55, (192, ), (1, ))
    assert_size_stride(primals_56, (192, ), (1, ))
    assert_size_stride(primals_57, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_61, (224, ), (1, ))
    assert_size_stride(primals_62, (224, ), (1, ))
    assert_size_stride(primals_63, (128, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_73, (288, ), (1, ))
    assert_size_stride(primals_74, (288, ), (1, ))
    assert_size_stride(primals_75, (128, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_78, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_79, (320, ), (1, ))
    assert_size_stride(primals_80, (320, ), (1, ))
    assert_size_stride(primals_81, (128, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_85, (352, ), (1, ))
    assert_size_stride(primals_86, (352, ), (1, ))
    assert_size_stride(primals_87, (128, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_91, (384, ), (1, ))
    assert_size_stride(primals_92, (384, ), (1, ))
    assert_size_stride(primals_93, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_96, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_97, (416, ), (1, ))
    assert_size_stride(primals_98, (416, ), (1, ))
    assert_size_stride(primals_99, (128, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_103, (448, ), (1, ))
    assert_size_stride(primals_104, (448, ), (1, ))
    assert_size_stride(primals_105, (128, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_109, (480, ), (1, ))
    assert_size_stride(primals_110, (480, ), (1, ))
    assert_size_stride(primals_111, (128, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_112, (128, ), (1, ))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_124, (288, ), (1, ))
    assert_size_stride(primals_125, (288, ), (1, ))
    assert_size_stride(primals_126, (128, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_130, (320, ), (1, ))
    assert_size_stride(primals_131, (320, ), (1, ))
    assert_size_stride(primals_132, (128, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_136, (352, ), (1, ))
    assert_size_stride(primals_137, (352, ), (1, ))
    assert_size_stride(primals_138, (128, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_141, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_142, (384, ), (1, ))
    assert_size_stride(primals_143, (384, ), (1, ))
    assert_size_stride(primals_144, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (128, ), (1, ))
    assert_size_stride(primals_147, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_148, (416, ), (1, ))
    assert_size_stride(primals_149, (416, ), (1, ))
    assert_size_stride(primals_150, (128, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_151, (128, ), (1, ))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_153, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_154, (448, ), (1, ))
    assert_size_stride(primals_155, (448, ), (1, ))
    assert_size_stride(primals_156, (128, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_157, (128, ), (1, ))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_160, (480, ), (1, ))
    assert_size_stride(primals_161, (480, ), (1, ))
    assert_size_stride(primals_162, (128, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, ), (1, ))
    assert_size_stride(primals_165, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (512, ), (1, ))
    assert_size_stride(primals_168, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_172, (544, ), (1, ))
    assert_size_stride(primals_173, (544, ), (1, ))
    assert_size_stride(primals_174, (128, 544, 1, 1), (544, 1, 1, 1))
    assert_size_stride(primals_175, (128, ), (1, ))
    assert_size_stride(primals_176, (128, ), (1, ))
    assert_size_stride(primals_177, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_178, (576, ), (1, ))
    assert_size_stride(primals_179, (576, ), (1, ))
    assert_size_stride(primals_180, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_183, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_184, (608, ), (1, ))
    assert_size_stride(primals_185, (608, ), (1, ))
    assert_size_stride(primals_186, (128, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_190, (640, ), (1, ))
    assert_size_stride(primals_191, (640, ), (1, ))
    assert_size_stride(primals_192, (128, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_193, (128, ), (1, ))
    assert_size_stride(primals_194, (128, ), (1, ))
    assert_size_stride(primals_195, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_196, (672, ), (1, ))
    assert_size_stride(primals_197, (672, ), (1, ))
    assert_size_stride(primals_198, (128, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_199, (128, ), (1, ))
    assert_size_stride(primals_200, (128, ), (1, ))
    assert_size_stride(primals_201, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_202, (704, ), (1, ))
    assert_size_stride(primals_203, (704, ), (1, ))
    assert_size_stride(primals_204, (128, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_205, (128, ), (1, ))
    assert_size_stride(primals_206, (128, ), (1, ))
    assert_size_stride(primals_207, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_208, (736, ), (1, ))
    assert_size_stride(primals_209, (736, ), (1, ))
    assert_size_stride(primals_210, (128, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_216, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_217, (128, ), (1, ))
    assert_size_stride(primals_218, (128, ), (1, ))
    assert_size_stride(primals_219, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_220, (800, ), (1, ))
    assert_size_stride(primals_221, (800, ), (1, ))
    assert_size_stride(primals_222, (128, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_223, (128, ), (1, ))
    assert_size_stride(primals_224, (128, ), (1, ))
    assert_size_stride(primals_225, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_226, (832, ), (1, ))
    assert_size_stride(primals_227, (832, ), (1, ))
    assert_size_stride(primals_228, (128, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (128, ), (1, ))
    assert_size_stride(primals_231, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_232, (864, ), (1, ))
    assert_size_stride(primals_233, (864, ), (1, ))
    assert_size_stride(primals_234, (128, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(primals_235, (128, ), (1, ))
    assert_size_stride(primals_236, (128, ), (1, ))
    assert_size_stride(primals_237, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_238, (896, ), (1, ))
    assert_size_stride(primals_239, (896, ), (1, ))
    assert_size_stride(primals_240, (128, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_241, (128, ), (1, ))
    assert_size_stride(primals_242, (128, ), (1, ))
    assert_size_stride(primals_243, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_244, (928, ), (1, ))
    assert_size_stride(primals_245, (928, ), (1, ))
    assert_size_stride(primals_246, (128, 928, 1, 1), (928, 1, 1, 1))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_248, (128, ), (1, ))
    assert_size_stride(primals_249, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_250, (960, ), (1, ))
    assert_size_stride(primals_251, (960, ), (1, ))
    assert_size_stride(primals_252, (128, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_253, (128, ), (1, ))
    assert_size_stride(primals_254, (128, ), (1, ))
    assert_size_stride(primals_255, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_256, (992, ), (1, ))
    assert_size_stride(primals_257, (992, ), (1, ))
    assert_size_stride(primals_258, (128, 992, 1, 1), (992, 1, 1, 1))
    assert_size_stride(primals_259, (128, ), (1, ))
    assert_size_stride(primals_260, (128, ), (1, ))
    assert_size_stride(primals_261, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_262, (1024, ), (1, ))
    assert_size_stride(primals_263, (1024, ), (1, ))
    assert_size_stride(primals_264, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_265, (512, ), (1, ))
    assert_size_stride(primals_266, (512, ), (1, ))
    assert_size_stride(primals_267, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_271, (544, ), (1, ))
    assert_size_stride(primals_272, (544, ), (1, ))
    assert_size_stride(primals_273, (128, 544, 1, 1), (544, 1, 1, 1))
    assert_size_stride(primals_274, (128, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_277, (576, ), (1, ))
    assert_size_stride(primals_278, (576, ), (1, ))
    assert_size_stride(primals_279, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_283, (608, ), (1, ))
    assert_size_stride(primals_284, (608, ), (1, ))
    assert_size_stride(primals_285, (128, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, ), (1, ))
    assert_size_stride(primals_288, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_289, (640, ), (1, ))
    assert_size_stride(primals_290, (640, ), (1, ))
    assert_size_stride(primals_291, (128, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_292, (128, ), (1, ))
    assert_size_stride(primals_293, (128, ), (1, ))
    assert_size_stride(primals_294, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_295, (672, ), (1, ))
    assert_size_stride(primals_296, (672, ), (1, ))
    assert_size_stride(primals_297, (128, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (128, ), (1, ))
    assert_size_stride(primals_300, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_301, (704, ), (1, ))
    assert_size_stride(primals_302, (704, ), (1, ))
    assert_size_stride(primals_303, (128, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_304, (128, ), (1, ))
    assert_size_stride(primals_305, (128, ), (1, ))
    assert_size_stride(primals_306, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_307, (736, ), (1, ))
    assert_size_stride(primals_308, (736, ), (1, ))
    assert_size_stride(primals_309, (128, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_310, (128, ), (1, ))
    assert_size_stride(primals_311, (128, ), (1, ))
    assert_size_stride(primals_312, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_313, (768, ), (1, ))
    assert_size_stride(primals_314, (768, ), (1, ))
    assert_size_stride(primals_315, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_316, (128, ), (1, ))
    assert_size_stride(primals_317, (128, ), (1, ))
    assert_size_stride(primals_318, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_319, (800, ), (1, ))
    assert_size_stride(primals_320, (800, ), (1, ))
    assert_size_stride(primals_321, (128, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_322, (128, ), (1, ))
    assert_size_stride(primals_323, (128, ), (1, ))
    assert_size_stride(primals_324, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_325, (832, ), (1, ))
    assert_size_stride(primals_326, (832, ), (1, ))
    assert_size_stride(primals_327, (128, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_328, (128, ), (1, ))
    assert_size_stride(primals_329, (128, ), (1, ))
    assert_size_stride(primals_330, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_331, (864, ), (1, ))
    assert_size_stride(primals_332, (864, ), (1, ))
    assert_size_stride(primals_333, (128, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(primals_334, (128, ), (1, ))
    assert_size_stride(primals_335, (128, ), (1, ))
    assert_size_stride(primals_336, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_337, (896, ), (1, ))
    assert_size_stride(primals_338, (896, ), (1, ))
    assert_size_stride(primals_339, (128, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_340, (128, ), (1, ))
    assert_size_stride(primals_341, (128, ), (1, ))
    assert_size_stride(primals_342, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_343, (928, ), (1, ))
    assert_size_stride(primals_344, (928, ), (1, ))
    assert_size_stride(primals_345, (128, 928, 1, 1), (928, 1, 1, 1))
    assert_size_stride(primals_346, (128, ), (1, ))
    assert_size_stride(primals_347, (128, ), (1, ))
    assert_size_stride(primals_348, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_349, (960, ), (1, ))
    assert_size_stride(primals_350, (960, ), (1, ))
    assert_size_stride(primals_351, (128, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_352, (128, ), (1, ))
    assert_size_stride(primals_353, (128, ), (1, ))
    assert_size_stride(primals_354, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_355, (992, ), (1, ))
    assert_size_stride(primals_356, (992, ), (1, ))
    assert_size_stride(primals_357, (128, 992, 1, 1), (992, 1, 1, 1))
    assert_size_stride(primals_358, (128, ), (1, ))
    assert_size_stride(primals_359, (128, ), (1, ))
    assert_size_stride(primals_360, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_361, (1024, ), (1, ))
    assert_size_stride(primals_362, (1024, ), (1, ))
    assert_size_stride(primals_363, (1000, 1024), (1024, 1))
    assert_size_stride(primals_364, (1000, ), (1, ))
    assert_size_stride(primals_365, (64, ), (1, ))
    assert_size_stride(primals_366, (64, ), (1, ))
    assert_size_stride(primals_367, (), ())
    assert_size_stride(primals_368, (64, ), (1, ))
    assert_size_stride(primals_369, (64, ), (1, ))
    assert_size_stride(primals_370, (), ())
    assert_size_stride(primals_371, (128, ), (1, ))
    assert_size_stride(primals_372, (128, ), (1, ))
    assert_size_stride(primals_373, (), ())
    assert_size_stride(primals_374, (96, ), (1, ))
    assert_size_stride(primals_375, (96, ), (1, ))
    assert_size_stride(primals_376, (), ())
    assert_size_stride(primals_377, (128, ), (1, ))
    assert_size_stride(primals_378, (128, ), (1, ))
    assert_size_stride(primals_379, (), ())
    assert_size_stride(primals_380, (128, ), (1, ))
    assert_size_stride(primals_381, (128, ), (1, ))
    assert_size_stride(primals_382, (), ())
    assert_size_stride(primals_383, (128, ), (1, ))
    assert_size_stride(primals_384, (128, ), (1, ))
    assert_size_stride(primals_385, (), ())
    assert_size_stride(primals_386, (160, ), (1, ))
    assert_size_stride(primals_387, (160, ), (1, ))
    assert_size_stride(primals_388, (), ())
    assert_size_stride(primals_389, (128, ), (1, ))
    assert_size_stride(primals_390, (128, ), (1, ))
    assert_size_stride(primals_391, (), ())
    assert_size_stride(primals_392, (192, ), (1, ))
    assert_size_stride(primals_393, (192, ), (1, ))
    assert_size_stride(primals_394, (), ())
    assert_size_stride(primals_395, (128, ), (1, ))
    assert_size_stride(primals_396, (128, ), (1, ))
    assert_size_stride(primals_397, (), ())
    assert_size_stride(primals_398, (224, ), (1, ))
    assert_size_stride(primals_399, (224, ), (1, ))
    assert_size_stride(primals_400, (), ())
    assert_size_stride(primals_401, (128, ), (1, ))
    assert_size_stride(primals_402, (128, ), (1, ))
    assert_size_stride(primals_403, (), ())
    assert_size_stride(primals_404, (256, ), (1, ))
    assert_size_stride(primals_405, (256, ), (1, ))
    assert_size_stride(primals_406, (), ())
    assert_size_stride(primals_407, (128, ), (1, ))
    assert_size_stride(primals_408, (128, ), (1, ))
    assert_size_stride(primals_409, (), ())
    assert_size_stride(primals_410, (128, ), (1, ))
    assert_size_stride(primals_411, (128, ), (1, ))
    assert_size_stride(primals_412, (), ())
    assert_size_stride(primals_413, (160, ), (1, ))
    assert_size_stride(primals_414, (160, ), (1, ))
    assert_size_stride(primals_415, (), ())
    assert_size_stride(primals_416, (128, ), (1, ))
    assert_size_stride(primals_417, (128, ), (1, ))
    assert_size_stride(primals_418, (), ())
    assert_size_stride(primals_419, (192, ), (1, ))
    assert_size_stride(primals_420, (192, ), (1, ))
    assert_size_stride(primals_421, (), ())
    assert_size_stride(primals_422, (128, ), (1, ))
    assert_size_stride(primals_423, (128, ), (1, ))
    assert_size_stride(primals_424, (), ())
    assert_size_stride(primals_425, (224, ), (1, ))
    assert_size_stride(primals_426, (224, ), (1, ))
    assert_size_stride(primals_427, (), ())
    assert_size_stride(primals_428, (128, ), (1, ))
    assert_size_stride(primals_429, (128, ), (1, ))
    assert_size_stride(primals_430, (), ())
    assert_size_stride(primals_431, (256, ), (1, ))
    assert_size_stride(primals_432, (256, ), (1, ))
    assert_size_stride(primals_433, (), ())
    assert_size_stride(primals_434, (128, ), (1, ))
    assert_size_stride(primals_435, (128, ), (1, ))
    assert_size_stride(primals_436, (), ())
    assert_size_stride(primals_437, (288, ), (1, ))
    assert_size_stride(primals_438, (288, ), (1, ))
    assert_size_stride(primals_439, (), ())
    assert_size_stride(primals_440, (128, ), (1, ))
    assert_size_stride(primals_441, (128, ), (1, ))
    assert_size_stride(primals_442, (), ())
    assert_size_stride(primals_443, (320, ), (1, ))
    assert_size_stride(primals_444, (320, ), (1, ))
    assert_size_stride(primals_445, (), ())
    assert_size_stride(primals_446, (128, ), (1, ))
    assert_size_stride(primals_447, (128, ), (1, ))
    assert_size_stride(primals_448, (), ())
    assert_size_stride(primals_449, (352, ), (1, ))
    assert_size_stride(primals_450, (352, ), (1, ))
    assert_size_stride(primals_451, (), ())
    assert_size_stride(primals_452, (128, ), (1, ))
    assert_size_stride(primals_453, (128, ), (1, ))
    assert_size_stride(primals_454, (), ())
    assert_size_stride(primals_455, (384, ), (1, ))
    assert_size_stride(primals_456, (384, ), (1, ))
    assert_size_stride(primals_457, (), ())
    assert_size_stride(primals_458, (128, ), (1, ))
    assert_size_stride(primals_459, (128, ), (1, ))
    assert_size_stride(primals_460, (), ())
    assert_size_stride(primals_461, (416, ), (1, ))
    assert_size_stride(primals_462, (416, ), (1, ))
    assert_size_stride(primals_463, (), ())
    assert_size_stride(primals_464, (128, ), (1, ))
    assert_size_stride(primals_465, (128, ), (1, ))
    assert_size_stride(primals_466, (), ())
    assert_size_stride(primals_467, (448, ), (1, ))
    assert_size_stride(primals_468, (448, ), (1, ))
    assert_size_stride(primals_469, (), ())
    assert_size_stride(primals_470, (128, ), (1, ))
    assert_size_stride(primals_471, (128, ), (1, ))
    assert_size_stride(primals_472, (), ())
    assert_size_stride(primals_473, (480, ), (1, ))
    assert_size_stride(primals_474, (480, ), (1, ))
    assert_size_stride(primals_475, (), ())
    assert_size_stride(primals_476, (128, ), (1, ))
    assert_size_stride(primals_477, (128, ), (1, ))
    assert_size_stride(primals_478, (), ())
    assert_size_stride(primals_479, (512, ), (1, ))
    assert_size_stride(primals_480, (512, ), (1, ))
    assert_size_stride(primals_481, (), ())
    assert_size_stride(primals_482, (256, ), (1, ))
    assert_size_stride(primals_483, (256, ), (1, ))
    assert_size_stride(primals_484, (), ())
    assert_size_stride(primals_485, (128, ), (1, ))
    assert_size_stride(primals_486, (128, ), (1, ))
    assert_size_stride(primals_487, (), ())
    assert_size_stride(primals_488, (288, ), (1, ))
    assert_size_stride(primals_489, (288, ), (1, ))
    assert_size_stride(primals_490, (), ())
    assert_size_stride(primals_491, (128, ), (1, ))
    assert_size_stride(primals_492, (128, ), (1, ))
    assert_size_stride(primals_493, (), ())
    assert_size_stride(primals_494, (320, ), (1, ))
    assert_size_stride(primals_495, (320, ), (1, ))
    assert_size_stride(primals_496, (), ())
    assert_size_stride(primals_497, (128, ), (1, ))
    assert_size_stride(primals_498, (128, ), (1, ))
    assert_size_stride(primals_499, (), ())
    assert_size_stride(primals_500, (352, ), (1, ))
    assert_size_stride(primals_501, (352, ), (1, ))
    assert_size_stride(primals_502, (), ())
    assert_size_stride(primals_503, (128, ), (1, ))
    assert_size_stride(primals_504, (128, ), (1, ))
    assert_size_stride(primals_505, (), ())
    assert_size_stride(primals_506, (384, ), (1, ))
    assert_size_stride(primals_507, (384, ), (1, ))
    assert_size_stride(primals_508, (), ())
    assert_size_stride(primals_509, (128, ), (1, ))
    assert_size_stride(primals_510, (128, ), (1, ))
    assert_size_stride(primals_511, (), ())
    assert_size_stride(primals_512, (416, ), (1, ))
    assert_size_stride(primals_513, (416, ), (1, ))
    assert_size_stride(primals_514, (), ())
    assert_size_stride(primals_515, (128, ), (1, ))
    assert_size_stride(primals_516, (128, ), (1, ))
    assert_size_stride(primals_517, (), ())
    assert_size_stride(primals_518, (448, ), (1, ))
    assert_size_stride(primals_519, (448, ), (1, ))
    assert_size_stride(primals_520, (), ())
    assert_size_stride(primals_521, (128, ), (1, ))
    assert_size_stride(primals_522, (128, ), (1, ))
    assert_size_stride(primals_523, (), ())
    assert_size_stride(primals_524, (480, ), (1, ))
    assert_size_stride(primals_525, (480, ), (1, ))
    assert_size_stride(primals_526, (), ())
    assert_size_stride(primals_527, (128, ), (1, ))
    assert_size_stride(primals_528, (128, ), (1, ))
    assert_size_stride(primals_529, (), ())
    assert_size_stride(primals_530, (512, ), (1, ))
    assert_size_stride(primals_531, (512, ), (1, ))
    assert_size_stride(primals_532, (), ())
    assert_size_stride(primals_533, (128, ), (1, ))
    assert_size_stride(primals_534, (128, ), (1, ))
    assert_size_stride(primals_535, (), ())
    assert_size_stride(primals_536, (544, ), (1, ))
    assert_size_stride(primals_537, (544, ), (1, ))
    assert_size_stride(primals_538, (), ())
    assert_size_stride(primals_539, (128, ), (1, ))
    assert_size_stride(primals_540, (128, ), (1, ))
    assert_size_stride(primals_541, (), ())
    assert_size_stride(primals_542, (576, ), (1, ))
    assert_size_stride(primals_543, (576, ), (1, ))
    assert_size_stride(primals_544, (), ())
    assert_size_stride(primals_545, (128, ), (1, ))
    assert_size_stride(primals_546, (128, ), (1, ))
    assert_size_stride(primals_547, (), ())
    assert_size_stride(primals_548, (608, ), (1, ))
    assert_size_stride(primals_549, (608, ), (1, ))
    assert_size_stride(primals_550, (), ())
    assert_size_stride(primals_551, (128, ), (1, ))
    assert_size_stride(primals_552, (128, ), (1, ))
    assert_size_stride(primals_553, (), ())
    assert_size_stride(primals_554, (640, ), (1, ))
    assert_size_stride(primals_555, (640, ), (1, ))
    assert_size_stride(primals_556, (), ())
    assert_size_stride(primals_557, (128, ), (1, ))
    assert_size_stride(primals_558, (128, ), (1, ))
    assert_size_stride(primals_559, (), ())
    assert_size_stride(primals_560, (672, ), (1, ))
    assert_size_stride(primals_561, (672, ), (1, ))
    assert_size_stride(primals_562, (), ())
    assert_size_stride(primals_563, (128, ), (1, ))
    assert_size_stride(primals_564, (128, ), (1, ))
    assert_size_stride(primals_565, (), ())
    assert_size_stride(primals_566, (704, ), (1, ))
    assert_size_stride(primals_567, (704, ), (1, ))
    assert_size_stride(primals_568, (), ())
    assert_size_stride(primals_569, (128, ), (1, ))
    assert_size_stride(primals_570, (128, ), (1, ))
    assert_size_stride(primals_571, (), ())
    assert_size_stride(primals_572, (736, ), (1, ))
    assert_size_stride(primals_573, (736, ), (1, ))
    assert_size_stride(primals_574, (), ())
    assert_size_stride(primals_575, (128, ), (1, ))
    assert_size_stride(primals_576, (128, ), (1, ))
    assert_size_stride(primals_577, (), ())
    assert_size_stride(primals_578, (768, ), (1, ))
    assert_size_stride(primals_579, (768, ), (1, ))
    assert_size_stride(primals_580, (), ())
    assert_size_stride(primals_581, (128, ), (1, ))
    assert_size_stride(primals_582, (128, ), (1, ))
    assert_size_stride(primals_583, (), ())
    assert_size_stride(primals_584, (800, ), (1, ))
    assert_size_stride(primals_585, (800, ), (1, ))
    assert_size_stride(primals_586, (), ())
    assert_size_stride(primals_587, (128, ), (1, ))
    assert_size_stride(primals_588, (128, ), (1, ))
    assert_size_stride(primals_589, (), ())
    assert_size_stride(primals_590, (832, ), (1, ))
    assert_size_stride(primals_591, (832, ), (1, ))
    assert_size_stride(primals_592, (), ())
    assert_size_stride(primals_593, (128, ), (1, ))
    assert_size_stride(primals_594, (128, ), (1, ))
    assert_size_stride(primals_595, (), ())
    assert_size_stride(primals_596, (864, ), (1, ))
    assert_size_stride(primals_597, (864, ), (1, ))
    assert_size_stride(primals_598, (), ())
    assert_size_stride(primals_599, (128, ), (1, ))
    assert_size_stride(primals_600, (128, ), (1, ))
    assert_size_stride(primals_601, (), ())
    assert_size_stride(primals_602, (896, ), (1, ))
    assert_size_stride(primals_603, (896, ), (1, ))
    assert_size_stride(primals_604, (), ())
    assert_size_stride(primals_605, (128, ), (1, ))
    assert_size_stride(primals_606, (128, ), (1, ))
    assert_size_stride(primals_607, (), ())
    assert_size_stride(primals_608, (928, ), (1, ))
    assert_size_stride(primals_609, (928, ), (1, ))
    assert_size_stride(primals_610, (), ())
    assert_size_stride(primals_611, (128, ), (1, ))
    assert_size_stride(primals_612, (128, ), (1, ))
    assert_size_stride(primals_613, (), ())
    assert_size_stride(primals_614, (960, ), (1, ))
    assert_size_stride(primals_615, (960, ), (1, ))
    assert_size_stride(primals_616, (), ())
    assert_size_stride(primals_617, (128, ), (1, ))
    assert_size_stride(primals_618, (128, ), (1, ))
    assert_size_stride(primals_619, (), ())
    assert_size_stride(primals_620, (992, ), (1, ))
    assert_size_stride(primals_621, (992, ), (1, ))
    assert_size_stride(primals_622, (), ())
    assert_size_stride(primals_623, (128, ), (1, ))
    assert_size_stride(primals_624, (128, ), (1, ))
    assert_size_stride(primals_625, (), ())
    assert_size_stride(primals_626, (1024, ), (1, ))
    assert_size_stride(primals_627, (1024, ), (1, ))
    assert_size_stride(primals_628, (), ())
    assert_size_stride(primals_629, (512, ), (1, ))
    assert_size_stride(primals_630, (512, ), (1, ))
    assert_size_stride(primals_631, (), ())
    assert_size_stride(primals_632, (128, ), (1, ))
    assert_size_stride(primals_633, (128, ), (1, ))
    assert_size_stride(primals_634, (), ())
    assert_size_stride(primals_635, (544, ), (1, ))
    assert_size_stride(primals_636, (544, ), (1, ))
    assert_size_stride(primals_637, (), ())
    assert_size_stride(primals_638, (128, ), (1, ))
    assert_size_stride(primals_639, (128, ), (1, ))
    assert_size_stride(primals_640, (), ())
    assert_size_stride(primals_641, (576, ), (1, ))
    assert_size_stride(primals_642, (576, ), (1, ))
    assert_size_stride(primals_643, (), ())
    assert_size_stride(primals_644, (128, ), (1, ))
    assert_size_stride(primals_645, (128, ), (1, ))
    assert_size_stride(primals_646, (), ())
    assert_size_stride(primals_647, (608, ), (1, ))
    assert_size_stride(primals_648, (608, ), (1, ))
    assert_size_stride(primals_649, (), ())
    assert_size_stride(primals_650, (128, ), (1, ))
    assert_size_stride(primals_651, (128, ), (1, ))
    assert_size_stride(primals_652, (), ())
    assert_size_stride(primals_653, (640, ), (1, ))
    assert_size_stride(primals_654, (640, ), (1, ))
    assert_size_stride(primals_655, (), ())
    assert_size_stride(primals_656, (128, ), (1, ))
    assert_size_stride(primals_657, (128, ), (1, ))
    assert_size_stride(primals_658, (), ())
    assert_size_stride(primals_659, (672, ), (1, ))
    assert_size_stride(primals_660, (672, ), (1, ))
    assert_size_stride(primals_661, (), ())
    assert_size_stride(primals_662, (128, ), (1, ))
    assert_size_stride(primals_663, (128, ), (1, ))
    assert_size_stride(primals_664, (), ())
    assert_size_stride(primals_665, (704, ), (1, ))
    assert_size_stride(primals_666, (704, ), (1, ))
    assert_size_stride(primals_667, (), ())
    assert_size_stride(primals_668, (128, ), (1, ))
    assert_size_stride(primals_669, (128, ), (1, ))
    assert_size_stride(primals_670, (), ())
    assert_size_stride(primals_671, (736, ), (1, ))
    assert_size_stride(primals_672, (736, ), (1, ))
    assert_size_stride(primals_673, (), ())
    assert_size_stride(primals_674, (128, ), (1, ))
    assert_size_stride(primals_675, (128, ), (1, ))
    assert_size_stride(primals_676, (), ())
    assert_size_stride(primals_677, (768, ), (1, ))
    assert_size_stride(primals_678, (768, ), (1, ))
    assert_size_stride(primals_679, (), ())
    assert_size_stride(primals_680, (128, ), (1, ))
    assert_size_stride(primals_681, (128, ), (1, ))
    assert_size_stride(primals_682, (), ())
    assert_size_stride(primals_683, (800, ), (1, ))
    assert_size_stride(primals_684, (800, ), (1, ))
    assert_size_stride(primals_685, (), ())
    assert_size_stride(primals_686, (128, ), (1, ))
    assert_size_stride(primals_687, (128, ), (1, ))
    assert_size_stride(primals_688, (), ())
    assert_size_stride(primals_689, (832, ), (1, ))
    assert_size_stride(primals_690, (832, ), (1, ))
    assert_size_stride(primals_691, (), ())
    assert_size_stride(primals_692, (128, ), (1, ))
    assert_size_stride(primals_693, (128, ), (1, ))
    assert_size_stride(primals_694, (), ())
    assert_size_stride(primals_695, (864, ), (1, ))
    assert_size_stride(primals_696, (864, ), (1, ))
    assert_size_stride(primals_697, (), ())
    assert_size_stride(primals_698, (128, ), (1, ))
    assert_size_stride(primals_699, (128, ), (1, ))
    assert_size_stride(primals_700, (), ())
    assert_size_stride(primals_701, (896, ), (1, ))
    assert_size_stride(primals_702, (896, ), (1, ))
    assert_size_stride(primals_703, (), ())
    assert_size_stride(primals_704, (128, ), (1, ))
    assert_size_stride(primals_705, (128, ), (1, ))
    assert_size_stride(primals_706, (), ())
    assert_size_stride(primals_707, (928, ), (1, ))
    assert_size_stride(primals_708, (928, ), (1, ))
    assert_size_stride(primals_709, (), ())
    assert_size_stride(primals_710, (128, ), (1, ))
    assert_size_stride(primals_711, (128, ), (1, ))
    assert_size_stride(primals_712, (), ())
    assert_size_stride(primals_713, (960, ), (1, ))
    assert_size_stride(primals_714, (960, ), (1, ))
    assert_size_stride(primals_715, (), ())
    assert_size_stride(primals_716, (128, ), (1, ))
    assert_size_stride(primals_717, (128, ), (1, ))
    assert_size_stride(primals_718, (), ())
    assert_size_stride(primals_719, (992, ), (1, ))
    assert_size_stride(primals_720, (992, ), (1, ))
    assert_size_stride(primals_721, (), ())
    assert_size_stride(primals_722, (128, ), (1, ))
    assert_size_stride(primals_723, (128, ), (1, ))
    assert_size_stride(primals_724, (), ())
    assert_size_stride(primals_725, (1024, ), (1, ))
    assert_size_stride(primals_726, (1024, ), (1, ))
    assert_size_stride(primals_727, (), ())
    assert_size_stride(primals_728, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf26 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf27 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf29 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf30 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf32 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf33 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf35 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf37 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf38 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf39 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf40 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf42 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf43 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf44 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf45 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf46 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf47 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf48 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf49 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf50 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf51 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf52 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf53 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf55 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf56 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf57 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf58 = empty_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf59 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(primals_728.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()))
    del primals_1
    del primals_102
    del primals_108
    del primals_114
    del primals_123
    del primals_129
    del primals_135
    del primals_141
    del primals_147
    del primals_15
    del primals_153
    del primals_159
    del primals_165
    del primals_171
    del primals_177
    del primals_183
    del primals_189
    del primals_195
    del primals_201
    del primals_207
    del primals_21
    del primals_213
    del primals_219
    del primals_225
    del primals_231
    del primals_237
    del primals_243
    del primals_249
    del primals_255
    del primals_261
    del primals_27
    del primals_270
    del primals_276
    del primals_282
    del primals_288
    del primals_294
    del primals_300
    del primals_306
    del primals_312
    del primals_318
    del primals_324
    del primals_33
    del primals_330
    del primals_336
    del primals_342
    del primals_348
    del primals_354
    del primals_360
    del primals_39
    del primals_48
    del primals_54
    del primals_60
    del primals_66
    del primals_72
    del primals_728
    del primals_78
    del primals_84
    del primals_9
    del primals_90
    del primals_96
    # Source Nodes: [l__mod___features_conv0], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(buf59, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (4, 64, 112, 112), (802816, 1, 7168, 64))
    buf61 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    buf69 = empty_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    buf62 = reinterpret_tensor(buf69, (4, 64, 56, 56), (301056, 1, 5376, 96), 0)  # alias
    buf63 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.int64)
    buf64 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    buf89 = empty_strided((4, 192, 56, 56), (602112, 1, 10752, 192), device='cpu', dtype=torch.float32)
    buf84 = reinterpret_tensor(buf89, (4, 64, 56, 56), (602112, 1, 10752, 192), 0)  # alias
    buf100 = empty_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    buf94 = reinterpret_tensor(buf100, (4, 64, 56, 56), (702464, 1, 12544, 224), 0)  # alias
    buf112 = empty_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    buf105 = reinterpret_tensor(buf112, (4, 64, 56, 56), (802816, 1, 14336, 256), 0)  # alias
    cpp_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_1(c_void_p(buf60.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf105.data_ptr()))
    del primals_3
    del primals_5
    # Source Nodes: [bottleneck_output], Original ATen: [aten.convolution]
    buf65 = extern_kernels.convolution(buf64, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf65, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf66 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_2(c_void_p(buf65.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf66.data_ptr()))
    del primals_8
    # Source Nodes: [new_features], Original ATen: [aten.convolution]
    buf67 = extern_kernels.convolution(buf66, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf67, (4, 32, 56, 56), (100352, 1, 1792, 32))
    buf68 = reinterpret_tensor(buf69, (4, 32, 56, 56), (301056, 1, 5376, 96), 64)  # alias
    buf85 = reinterpret_tensor(buf89, (4, 32, 56, 56), (602112, 1, 10752, 192), 64)  # alias
    buf95 = reinterpret_tensor(buf100, (4, 32, 56, 56), (702464, 1, 12544, 224), 64)  # alias
    buf106 = reinterpret_tensor(buf112, (4, 32, 56, 56), (802816, 1, 14336, 256), 64)  # alias
    buf70 = empty_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_3(c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(primals_375.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf70.data_ptr()))
    del primals_11
    # Source Nodes: [bottleneck_output_2], Original ATen: [aten.convolution]
    buf71 = extern_kernels.convolution(buf70, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf71, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf72 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_4(c_void_p(buf71.data_ptr()), c_void_p(primals_377.data_ptr()), c_void_p(primals_378.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf72.data_ptr()))
    del primals_14
    # Source Nodes: [new_features_2], Original ATen: [aten.convolution]
    buf73 = extern_kernels.convolution(buf72, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf73, (4, 32, 56, 56), (100352, 1, 1792, 32))
    buf74 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf75 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_5(c_void_p(buf62.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del primals_17
    # Source Nodes: [bottleneck_output_4], Original ATen: [aten.convolution]
    buf76 = extern_kernels.convolution(buf75, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf76, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf77 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_6(c_void_p(buf76.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(primals_384.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf77.data_ptr()))
    del primals_20
    # Source Nodes: [new_features_4], Original ATen: [aten.convolution]
    buf78 = extern_kernels.convolution(buf77, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf78, (4, 32, 56, 56), (100352, 1, 1792, 32))
    buf79 = empty_strided((4, 160, 56, 56), (501760, 1, 8960, 160), device='cpu', dtype=torch.float32)
    buf80 = empty_strided((4, 160, 56, 56), (501760, 1, 8960, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_7(c_void_p(buf62.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()))
    del primals_23
    # Source Nodes: [bottleneck_output_6], Original ATen: [aten.convolution]
    buf81 = extern_kernels.convolution(buf80, primals_24, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf81, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf82 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_8(c_void_p(buf81.data_ptr()), c_void_p(primals_389.data_ptr()), c_void_p(primals_390.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf82.data_ptr()))
    del primals_26
    # Source Nodes: [new_features_6], Original ATen: [aten.convolution]
    buf83 = extern_kernels.convolution(buf82, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf83, (4, 32, 56, 56), (100352, 1, 1792, 32))
    buf86 = reinterpret_tensor(buf89, (4, 32, 56, 56), (602112, 1, 10752, 192), 96)  # alias
    buf96 = reinterpret_tensor(buf100, (4, 32, 56, 56), (702464, 1, 12544, 224), 96)  # alias
    buf107 = reinterpret_tensor(buf112, (4, 32, 56, 56), (802816, 1, 14336, 256), 96)  # alias
    buf87 = reinterpret_tensor(buf89, (4, 32, 56, 56), (602112, 1, 10752, 192), 128)  # alias
    buf97 = reinterpret_tensor(buf100, (4, 32, 56, 56), (702464, 1, 12544, 224), 128)  # alias
    buf108 = reinterpret_tensor(buf112, (4, 32, 56, 56), (802816, 1, 14336, 256), 128)  # alias
    buf88 = reinterpret_tensor(buf89, (4, 32, 56, 56), (602112, 1, 10752, 192), 160)  # alias
    buf98 = reinterpret_tensor(buf100, (4, 32, 56, 56), (702464, 1, 12544, 224), 160)  # alias
    buf109 = reinterpret_tensor(buf112, (4, 32, 56, 56), (802816, 1, 14336, 256), 160)  # alias
    buf90 = empty_strided((4, 192, 56, 56), (602112, 1, 10752, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_9(c_void_p(buf73.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf90.data_ptr()))
    del primals_29
    # Source Nodes: [bottleneck_output_8], Original ATen: [aten.convolution]
    buf91 = extern_kernels.convolution(buf90, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf91, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf92 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_10(c_void_p(buf91.data_ptr()), c_void_p(primals_395.data_ptr()), c_void_p(primals_396.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf92.data_ptr()))
    del primals_32
    # Source Nodes: [new_features_8], Original ATen: [aten.convolution]
    buf93 = extern_kernels.convolution(buf92, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf93, (4, 32, 56, 56), (100352, 1, 1792, 32))
    buf99 = reinterpret_tensor(buf100, (4, 32, 56, 56), (702464, 1, 12544, 224), 192)  # alias
    buf110 = reinterpret_tensor(buf112, (4, 32, 56, 56), (802816, 1, 14336, 256), 192)  # alias
    buf101 = empty_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_11(c_void_p(buf93.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(primals_398.data_ptr()), c_void_p(primals_399.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf101.data_ptr()))
    del primals_35
    # Source Nodes: [bottleneck_output_10], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf103 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_12(c_void_p(buf102.data_ptr()), c_void_p(primals_401.data_ptr()), c_void_p(primals_402.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf103.data_ptr()))
    del primals_38
    # Source Nodes: [new_features_10], Original ATen: [aten.convolution]
    buf104 = extern_kernels.convolution(buf103, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf104, (4, 32, 56, 56), (100352, 1, 1792, 32))
    buf111 = reinterpret_tensor(buf112, (4, 32, 56, 56), (802816, 1, 14336, 256), 224)  # alias
    buf113 = empty_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_13(c_void_p(buf104.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_404.data_ptr()), c_void_p(primals_405.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()))
    del primals_41
    # Source Nodes: [l__mod___features_transition1_conv], Original ATen: [aten.convolution]
    buf114 = extern_kernels.convolution(buf113, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf115 = reinterpret_tensor(buf104, (4, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf104  # reuse
    buf116 = reinterpret_tensor(buf93, (4, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf93  # reuse
    cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_14(c_void_p(buf114.data_ptr()), c_void_p(primals_407.data_ptr()), c_void_p(primals_408.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    del primals_44
    # Source Nodes: [bottleneck_output_12], Original ATen: [aten.convolution]
    buf117 = extern_kernels.convolution(buf116, primals_45, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf117, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf118 = reinterpret_tensor(buf83, (4, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf83  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_15(c_void_p(buf117.data_ptr()), c_void_p(primals_410.data_ptr()), c_void_p(primals_411.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf118.data_ptr()))
    del primals_47
    # Source Nodes: [new_features_12], Original ATen: [aten.convolution]
    buf119 = extern_kernels.convolution(buf118, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf119, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf120 = empty_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    buf121 = empty_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_16(c_void_p(buf115.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(primals_413.data_ptr()), c_void_p(primals_414.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    del primals_50
    # Source Nodes: [bottleneck_output_14], Original ATen: [aten.convolution]
    buf122 = extern_kernels.convolution(buf121, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf122, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf123 = reinterpret_tensor(buf78, (4, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_17(c_void_p(buf122.data_ptr()), c_void_p(primals_416.data_ptr()), c_void_p(primals_417.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf123.data_ptr()))
    del primals_53
    # Source Nodes: [new_features_14], Original ATen: [aten.convolution]
    buf124 = extern_kernels.convolution(buf123, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf125 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    buf126 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_18(c_void_p(buf115.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(primals_419.data_ptr()), c_void_p(primals_420.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    del primals_56
    # Source Nodes: [bottleneck_output_16], Original ATen: [aten.convolution]
    buf127 = extern_kernels.convolution(buf126, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf127, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf128 = reinterpret_tensor(buf73, (4, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf73  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_19(c_void_p(buf127.data_ptr()), c_void_p(primals_422.data_ptr()), c_void_p(primals_423.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf128.data_ptr()))
    del primals_59
    # Source Nodes: [new_features_16], Original ATen: [aten.convolution]
    buf129 = extern_kernels.convolution(buf128, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf129, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf130 = empty_strided((4, 224, 28, 28), (175616, 1, 6272, 224), device='cpu', dtype=torch.float32)
    buf131 = empty_strided((4, 224, 28, 28), (175616, 1, 6272, 224), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_20(c_void_p(buf115.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(primals_425.data_ptr()), c_void_p(primals_426.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del primals_62
    # Source Nodes: [bottleneck_output_18], Original ATen: [aten.convolution]
    buf132 = extern_kernels.convolution(buf131, primals_63, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf132, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf133 = reinterpret_tensor(buf67, (4, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf67  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_21(c_void_p(buf132.data_ptr()), c_void_p(primals_428.data_ptr()), c_void_p(primals_429.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf133.data_ptr()))
    del primals_65
    # Source Nodes: [new_features_18], Original ATen: [aten.convolution]
    buf134 = extern_kernels.convolution(buf133, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf134, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf140 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    buf135 = reinterpret_tensor(buf140, (4, 128, 28, 28), (200704, 1, 7168, 256), 0)  # alias
    buf151 = empty_strided((4, 288, 28, 28), (225792, 1, 8064, 288), device='cpu', dtype=torch.float32)
    buf145 = reinterpret_tensor(buf151, (4, 128, 28, 28), (225792, 1, 8064, 288), 0)  # alias
    buf163 = empty_strided((4, 320, 28, 28), (250880, 1, 8960, 320), device='cpu', dtype=torch.float32)
    buf156 = reinterpret_tensor(buf163, (4, 128, 28, 28), (250880, 1, 8960, 320), 0)  # alias
    buf176 = empty_strided((4, 352, 28, 28), (275968, 1, 9856, 352), device='cpu', dtype=torch.float32)
    buf168 = reinterpret_tensor(buf176, (4, 128, 28, 28), (275968, 1, 9856, 352), 0)  # alias
    buf190 = empty_strided((4, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    buf181 = reinterpret_tensor(buf190, (4, 128, 28, 28), (301056, 1, 10752, 384), 0)  # alias
    buf205 = empty_strided((4, 416, 28, 28), (326144, 1, 11648, 416), device='cpu', dtype=torch.float32)
    buf195 = reinterpret_tensor(buf205, (4, 128, 28, 28), (326144, 1, 11648, 416), 0)  # alias
    buf136 = reinterpret_tensor(buf140, (4, 32, 28, 28), (200704, 1, 7168, 256), 128)  # alias
    buf146 = reinterpret_tensor(buf151, (4, 32, 28, 28), (225792, 1, 8064, 288), 128)  # alias
    buf157 = reinterpret_tensor(buf163, (4, 32, 28, 28), (250880, 1, 8960, 320), 128)  # alias
    buf169 = reinterpret_tensor(buf176, (4, 32, 28, 28), (275968, 1, 9856, 352), 128)  # alias
    buf182 = reinterpret_tensor(buf190, (4, 32, 28, 28), (301056, 1, 10752, 384), 128)  # alias
    buf196 = reinterpret_tensor(buf205, (4, 32, 28, 28), (326144, 1, 11648, 416), 128)  # alias
    buf137 = reinterpret_tensor(buf140, (4, 32, 28, 28), (200704, 1, 7168, 256), 160)  # alias
    buf147 = reinterpret_tensor(buf151, (4, 32, 28, 28), (225792, 1, 8064, 288), 160)  # alias
    buf158 = reinterpret_tensor(buf163, (4, 32, 28, 28), (250880, 1, 8960, 320), 160)  # alias
    buf170 = reinterpret_tensor(buf176, (4, 32, 28, 28), (275968, 1, 9856, 352), 160)  # alias
    buf183 = reinterpret_tensor(buf190, (4, 32, 28, 28), (301056, 1, 10752, 384), 160)  # alias
    buf197 = reinterpret_tensor(buf205, (4, 32, 28, 28), (326144, 1, 11648, 416), 160)  # alias
    buf138 = reinterpret_tensor(buf140, (4, 32, 28, 28), (200704, 1, 7168, 256), 192)  # alias
    buf148 = reinterpret_tensor(buf151, (4, 32, 28, 28), (225792, 1, 8064, 288), 192)  # alias
    buf159 = reinterpret_tensor(buf163, (4, 32, 28, 28), (250880, 1, 8960, 320), 192)  # alias
    buf171 = reinterpret_tensor(buf176, (4, 32, 28, 28), (275968, 1, 9856, 352), 192)  # alias
    buf184 = reinterpret_tensor(buf190, (4, 32, 28, 28), (301056, 1, 10752, 384), 192)  # alias
    buf198 = reinterpret_tensor(buf205, (4, 32, 28, 28), (326144, 1, 11648, 416), 192)  # alias
    buf139 = reinterpret_tensor(buf140, (4, 32, 28, 28), (200704, 1, 7168, 256), 224)  # alias
    buf149 = reinterpret_tensor(buf151, (4, 32, 28, 28), (225792, 1, 8064, 288), 224)  # alias
    buf160 = reinterpret_tensor(buf163, (4, 32, 28, 28), (250880, 1, 8960, 320), 224)  # alias
    buf172 = reinterpret_tensor(buf176, (4, 32, 28, 28), (275968, 1, 9856, 352), 224)  # alias
    buf185 = reinterpret_tensor(buf190, (4, 32, 28, 28), (301056, 1, 10752, 384), 224)  # alias
    buf199 = reinterpret_tensor(buf205, (4, 32, 28, 28), (326144, 1, 11648, 416), 224)  # alias
    buf141 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_22(c_void_p(buf115.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_431.data_ptr()), c_void_p(primals_432.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf141.data_ptr()))
    del primals_68
    # Source Nodes: [bottleneck_output_20], Original ATen: [aten.convolution]
    buf142 = extern_kernels.convolution(buf141, primals_69, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf142, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf143 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_23(c_void_p(buf142.data_ptr()), c_void_p(primals_434.data_ptr()), c_void_p(primals_435.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf143.data_ptr()))
    del primals_71
    # Source Nodes: [new_features_20], Original ATen: [aten.convolution]
    buf144 = extern_kernels.convolution(buf143, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf144, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf150 = reinterpret_tensor(buf151, (4, 32, 28, 28), (225792, 1, 8064, 288), 256)  # alias
    buf161 = reinterpret_tensor(buf163, (4, 32, 28, 28), (250880, 1, 8960, 320), 256)  # alias
    buf173 = reinterpret_tensor(buf176, (4, 32, 28, 28), (275968, 1, 9856, 352), 256)  # alias
    buf186 = reinterpret_tensor(buf190, (4, 32, 28, 28), (301056, 1, 10752, 384), 256)  # alias
    buf200 = reinterpret_tensor(buf205, (4, 32, 28, 28), (326144, 1, 11648, 416), 256)  # alias
    buf152 = empty_strided((4, 288, 28, 28), (225792, 1, 8064, 288), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_24(c_void_p(buf144.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(primals_437.data_ptr()), c_void_p(primals_438.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf152.data_ptr()))
    del primals_74
    # Source Nodes: [bottleneck_output_22], Original ATen: [aten.convolution]
    buf153 = extern_kernels.convolution(buf152, primals_75, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf153, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf154 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_25(c_void_p(buf153.data_ptr()), c_void_p(primals_440.data_ptr()), c_void_p(primals_441.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf154.data_ptr()))
    del primals_77
    # Source Nodes: [new_features_22], Original ATen: [aten.convolution]
    buf155 = extern_kernels.convolution(buf154, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf155, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf162 = reinterpret_tensor(buf163, (4, 32, 28, 28), (250880, 1, 8960, 320), 288)  # alias
    buf174 = reinterpret_tensor(buf176, (4, 32, 28, 28), (275968, 1, 9856, 352), 288)  # alias
    buf187 = reinterpret_tensor(buf190, (4, 32, 28, 28), (301056, 1, 10752, 384), 288)  # alias
    buf201 = reinterpret_tensor(buf205, (4, 32, 28, 28), (326144, 1, 11648, 416), 288)  # alias
    buf221 = empty_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    buf216 = reinterpret_tensor(buf221, (4, 32, 28, 28), (351232, 1, 12544, 448), 288)  # alias
    buf164 = empty_strided((4, 320, 28, 28), (250880, 1, 8960, 320), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_26(c_void_p(buf155.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(primals_443.data_ptr()), c_void_p(primals_444.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf164.data_ptr()))
    del primals_80
    # Source Nodes: [bottleneck_output_24], Original ATen: [aten.convolution]
    buf165 = extern_kernels.convolution(buf164, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf165, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf166 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_27(c_void_p(buf165.data_ptr()), c_void_p(primals_446.data_ptr()), c_void_p(primals_447.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf166.data_ptr()))
    del primals_83
    # Source Nodes: [new_features_24], Original ATen: [aten.convolution]
    buf167 = extern_kernels.convolution(buf166, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf167, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf175 = reinterpret_tensor(buf176, (4, 32, 28, 28), (275968, 1, 9856, 352), 320)  # alias
    buf188 = reinterpret_tensor(buf190, (4, 32, 28, 28), (301056, 1, 10752, 384), 320)  # alias
    buf202 = reinterpret_tensor(buf205, (4, 32, 28, 28), (326144, 1, 11648, 416), 320)  # alias
    buf217 = reinterpret_tensor(buf221, (4, 32, 28, 28), (351232, 1, 12544, 448), 320)  # alias
    buf238 = empty_strided((4, 480, 28, 28), (376320, 1, 13440, 480), device='cpu', dtype=torch.float32)
    buf233 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 320)  # alias
    buf177 = empty_strided((4, 352, 28, 28), (275968, 1, 9856, 352), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_28(c_void_p(buf167.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(primals_449.data_ptr()), c_void_p(primals_450.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf177.data_ptr()))
    del primals_86
    # Source Nodes: [bottleneck_output_26], Original ATen: [aten.convolution]
    buf178 = extern_kernels.convolution(buf177, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf178, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf179 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_29(c_void_p(buf178.data_ptr()), c_void_p(primals_452.data_ptr()), c_void_p(primals_453.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf179.data_ptr()))
    del primals_89
    # Source Nodes: [new_features_26], Original ATen: [aten.convolution]
    buf180 = extern_kernels.convolution(buf179, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf180, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf189 = reinterpret_tensor(buf190, (4, 32, 28, 28), (301056, 1, 10752, 384), 352)  # alias
    buf203 = reinterpret_tensor(buf205, (4, 32, 28, 28), (326144, 1, 11648, 416), 352)  # alias
    buf218 = reinterpret_tensor(buf221, (4, 32, 28, 28), (351232, 1, 12544, 448), 352)  # alias
    buf234 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 352)  # alias
    buf256 = empty_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    buf251 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 352)  # alias
    buf191 = empty_strided((4, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_30(c_void_p(buf180.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(primals_455.data_ptr()), c_void_p(primals_456.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf191.data_ptr()))
    del primals_92
    # Source Nodes: [bottleneck_output_28], Original ATen: [aten.convolution]
    buf192 = extern_kernels.convolution(buf191, primals_93, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf192, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf193 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_31(c_void_p(buf192.data_ptr()), c_void_p(primals_458.data_ptr()), c_void_p(primals_459.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf193.data_ptr()))
    del primals_95
    # Source Nodes: [new_features_28], Original ATen: [aten.convolution]
    buf194 = extern_kernels.convolution(buf193, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf194, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf204 = reinterpret_tensor(buf205, (4, 32, 28, 28), (326144, 1, 11648, 416), 384)  # alias
    buf219 = reinterpret_tensor(buf221, (4, 32, 28, 28), (351232, 1, 12544, 448), 384)  # alias
    buf235 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 384)  # alias
    buf252 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 384)  # alias
    buf206 = empty_strided((4, 416, 28, 28), (326144, 1, 11648, 416), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_32(c_void_p(buf194.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(primals_461.data_ptr()), c_void_p(primals_462.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf206.data_ptr()))
    del primals_98
    # Source Nodes: [bottleneck_output_30], Original ATen: [aten.convolution]
    buf207 = extern_kernels.convolution(buf206, primals_99, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf207, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf208 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_33(c_void_p(buf207.data_ptr()), c_void_p(primals_464.data_ptr()), c_void_p(primals_465.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf208.data_ptr()))
    del primals_101
    # Source Nodes: [new_features_30], Original ATen: [aten.convolution]
    buf209 = extern_kernels.convolution(buf208, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf209, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf210 = reinterpret_tensor(buf221, (4, 128, 28, 28), (351232, 1, 12544, 448), 0)  # alias
    buf226 = reinterpret_tensor(buf238, (4, 128, 28, 28), (376320, 1, 13440, 480), 0)  # alias
    buf243 = reinterpret_tensor(buf256, (4, 128, 28, 28), (401408, 1, 14336, 512), 0)  # alias
    buf211 = reinterpret_tensor(buf221, (4, 32, 28, 28), (351232, 1, 12544, 448), 128)  # alias
    buf227 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 128)  # alias
    buf244 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 128)  # alias
    buf212 = reinterpret_tensor(buf221, (4, 32, 28, 28), (351232, 1, 12544, 448), 160)  # alias
    buf228 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 160)  # alias
    buf245 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 160)  # alias
    buf213 = reinterpret_tensor(buf221, (4, 32, 28, 28), (351232, 1, 12544, 448), 192)  # alias
    buf229 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 192)  # alias
    buf246 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 192)  # alias
    buf214 = reinterpret_tensor(buf221, (4, 32, 28, 28), (351232, 1, 12544, 448), 224)  # alias
    buf230 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 224)  # alias
    buf247 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 224)  # alias
    buf215 = reinterpret_tensor(buf221, (4, 32, 28, 28), (351232, 1, 12544, 448), 256)  # alias
    buf231 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 256)  # alias
    buf248 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 256)  # alias
    buf220 = reinterpret_tensor(buf221, (4, 32, 28, 28), (351232, 1, 12544, 448), 416)  # alias
    buf236 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 416)  # alias
    buf253 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 416)  # alias
    buf222 = empty_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_34(c_void_p(buf115.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(primals_467.data_ptr()), c_void_p(primals_468.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf222.data_ptr()))
    del primals_104
    # Source Nodes: [bottleneck_output_32], Original ATen: [aten.convolution]
    buf223 = extern_kernels.convolution(buf222, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf223, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf224 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_35(c_void_p(buf223.data_ptr()), c_void_p(primals_470.data_ptr()), c_void_p(primals_471.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf224.data_ptr()))
    del primals_107
    # Source Nodes: [new_features_32], Original ATen: [aten.convolution]
    buf225 = extern_kernels.convolution(buf224, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf225, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf232 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 288)  # alias
    buf249 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 288)  # alias
    buf237 = reinterpret_tensor(buf238, (4, 32, 28, 28), (376320, 1, 13440, 480), 448)  # alias
    buf254 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 448)  # alias
    buf239 = empty_strided((4, 480, 28, 28), (376320, 1, 13440, 480), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_36(c_void_p(buf155.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(primals_473.data_ptr()), c_void_p(primals_474.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf239.data_ptr()))
    del primals_110
    # Source Nodes: [bottleneck_output_34], Original ATen: [aten.convolution]
    buf240 = extern_kernels.convolution(buf239, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf240, (4, 128, 28, 28), (100352, 1, 3584, 128))
    buf241 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_37(c_void_p(buf240.data_ptr()), c_void_p(primals_476.data_ptr()), c_void_p(primals_477.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf241.data_ptr()))
    del primals_113
    # Source Nodes: [new_features_34], Original ATen: [aten.convolution]
    buf242 = extern_kernels.convolution(buf241, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf242, (4, 32, 28, 28), (25088, 1, 896, 32))
    buf250 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 320)  # alias
    buf255 = reinterpret_tensor(buf256, (4, 32, 28, 28), (401408, 1, 14336, 512), 480)  # alias
    buf257 = empty_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_38(c_void_p(buf167.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(primals_479.data_ptr()), c_void_p(primals_480.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()))
    del primals_116
    # Source Nodes: [l__mod___features_transition2_conv], Original ATen: [aten.convolution]
    buf258 = extern_kernels.convolution(buf257, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf258, (4, 256, 28, 28), (200704, 1, 7168, 256))
    buf259 = empty_strided((4, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    buf260 = empty_strided((4, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_39(c_void_p(buf258.data_ptr()), c_void_p(primals_482.data_ptr()), c_void_p(primals_483.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del primals_119
    # Source Nodes: [bottleneck_output_36], Original ATen: [aten.convolution]
    buf261 = extern_kernels.convolution(buf260, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf261, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf262 = reinterpret_tensor(buf242, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf242  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_40(c_void_p(buf261.data_ptr()), c_void_p(primals_485.data_ptr()), c_void_p(primals_486.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf262.data_ptr()))
    del primals_122
    # Source Nodes: [new_features_36], Original ATen: [aten.convolution]
    buf263 = extern_kernels.convolution(buf262, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf263, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf264 = empty_strided((4, 288, 14, 14), (56448, 1, 4032, 288), device='cpu', dtype=torch.float32)
    buf265 = empty_strided((4, 288, 14, 14), (56448, 1, 4032, 288), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_41(c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(primals_488.data_ptr()), c_void_p(primals_489.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del primals_125
    # Source Nodes: [bottleneck_output_38], Original ATen: [aten.convolution]
    buf266 = extern_kernels.convolution(buf265, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf266, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf267 = reinterpret_tensor(buf167, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf167  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_42(c_void_p(buf266.data_ptr()), c_void_p(primals_491.data_ptr()), c_void_p(primals_492.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf267.data_ptr()))
    del primals_128
    # Source Nodes: [new_features_38], Original ATen: [aten.convolution]
    buf268 = extern_kernels.convolution(buf267, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf268, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf269 = empty_strided((4, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    buf270 = empty_strided((4, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_43(c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(primals_494.data_ptr()), c_void_p(primals_495.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    del primals_131
    # Source Nodes: [bottleneck_output_40], Original ATen: [aten.convolution]
    buf271 = extern_kernels.convolution(buf270, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf271, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf272 = reinterpret_tensor(buf225, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf225  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_44(c_void_p(buf271.data_ptr()), c_void_p(primals_497.data_ptr()), c_void_p(primals_498.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf272.data_ptr()))
    del primals_134
    # Source Nodes: [new_features_40], Original ATen: [aten.convolution]
    buf273 = extern_kernels.convolution(buf272, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf273, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf274 = empty_strided((4, 352, 14, 14), (68992, 1, 4928, 352), device='cpu', dtype=torch.float32)
    buf275 = empty_strided((4, 352, 14, 14), (68992, 1, 4928, 352), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_45(c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(primals_500.data_ptr()), c_void_p(primals_501.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()))
    del primals_137
    # Source Nodes: [bottleneck_output_42], Original ATen: [aten.convolution]
    buf276 = extern_kernels.convolution(buf275, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf276, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf277 = reinterpret_tensor(buf155, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf155  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_46(c_void_p(buf276.data_ptr()), c_void_p(primals_503.data_ptr()), c_void_p(primals_504.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf277.data_ptr()))
    del primals_140
    # Source Nodes: [new_features_42], Original ATen: [aten.convolution]
    buf278 = extern_kernels.convolution(buf277, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf278, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf284 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf279 = reinterpret_tensor(buf284, (4, 256, 14, 14), (75264, 1, 5376, 384), 0)  # alias
    buf295 = empty_strided((4, 416, 14, 14), (81536, 1, 5824, 416), device='cpu', dtype=torch.float32)
    buf289 = reinterpret_tensor(buf295, (4, 256, 14, 14), (81536, 1, 5824, 416), 0)  # alias
    buf307 = empty_strided((4, 448, 14, 14), (87808, 1, 6272, 448), device='cpu', dtype=torch.float32)
    buf300 = reinterpret_tensor(buf307, (4, 256, 14, 14), (87808, 1, 6272, 448), 0)  # alias
    buf320 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    buf312 = reinterpret_tensor(buf320, (4, 256, 14, 14), (94080, 1, 6720, 480), 0)  # alias
    buf334 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    buf325 = reinterpret_tensor(buf334, (4, 256, 14, 14), (100352, 1, 7168, 512), 0)  # alias
    buf349 = empty_strided((4, 544, 14, 14), (106624, 1, 7616, 544), device='cpu', dtype=torch.float32)
    buf339 = reinterpret_tensor(buf349, (4, 256, 14, 14), (106624, 1, 7616, 544), 0)  # alias
    buf280 = reinterpret_tensor(buf284, (4, 32, 14, 14), (75264, 1, 5376, 384), 256)  # alias
    buf290 = reinterpret_tensor(buf295, (4, 32, 14, 14), (81536, 1, 5824, 416), 256)  # alias
    buf301 = reinterpret_tensor(buf307, (4, 32, 14, 14), (87808, 1, 6272, 448), 256)  # alias
    buf313 = reinterpret_tensor(buf320, (4, 32, 14, 14), (94080, 1, 6720, 480), 256)  # alias
    buf326 = reinterpret_tensor(buf334, (4, 32, 14, 14), (100352, 1, 7168, 512), 256)  # alias
    buf340 = reinterpret_tensor(buf349, (4, 32, 14, 14), (106624, 1, 7616, 544), 256)  # alias
    buf281 = reinterpret_tensor(buf284, (4, 32, 14, 14), (75264, 1, 5376, 384), 288)  # alias
    buf291 = reinterpret_tensor(buf295, (4, 32, 14, 14), (81536, 1, 5824, 416), 288)  # alias
    buf302 = reinterpret_tensor(buf307, (4, 32, 14, 14), (87808, 1, 6272, 448), 288)  # alias
    buf314 = reinterpret_tensor(buf320, (4, 32, 14, 14), (94080, 1, 6720, 480), 288)  # alias
    buf327 = reinterpret_tensor(buf334, (4, 32, 14, 14), (100352, 1, 7168, 512), 288)  # alias
    buf341 = reinterpret_tensor(buf349, (4, 32, 14, 14), (106624, 1, 7616, 544), 288)  # alias
    buf282 = reinterpret_tensor(buf284, (4, 32, 14, 14), (75264, 1, 5376, 384), 320)  # alias
    buf292 = reinterpret_tensor(buf295, (4, 32, 14, 14), (81536, 1, 5824, 416), 320)  # alias
    buf303 = reinterpret_tensor(buf307, (4, 32, 14, 14), (87808, 1, 6272, 448), 320)  # alias
    buf315 = reinterpret_tensor(buf320, (4, 32, 14, 14), (94080, 1, 6720, 480), 320)  # alias
    buf328 = reinterpret_tensor(buf334, (4, 32, 14, 14), (100352, 1, 7168, 512), 320)  # alias
    buf342 = reinterpret_tensor(buf349, (4, 32, 14, 14), (106624, 1, 7616, 544), 320)  # alias
    buf283 = reinterpret_tensor(buf284, (4, 32, 14, 14), (75264, 1, 5376, 384), 352)  # alias
    buf293 = reinterpret_tensor(buf295, (4, 32, 14, 14), (81536, 1, 5824, 416), 352)  # alias
    buf304 = reinterpret_tensor(buf307, (4, 32, 14, 14), (87808, 1, 6272, 448), 352)  # alias
    buf316 = reinterpret_tensor(buf320, (4, 32, 14, 14), (94080, 1, 6720, 480), 352)  # alias
    buf329 = reinterpret_tensor(buf334, (4, 32, 14, 14), (100352, 1, 7168, 512), 352)  # alias
    buf343 = reinterpret_tensor(buf349, (4, 32, 14, 14), (106624, 1, 7616, 544), 352)  # alias
    buf285 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_47(c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(primals_506.data_ptr()), c_void_p(primals_507.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf285.data_ptr()))
    del primals_143
    # Source Nodes: [bottleneck_output_44], Original ATen: [aten.convolution]
    buf286 = extern_kernels.convolution(buf285, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf286, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf287 = reinterpret_tensor(buf209, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf209  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_48(c_void_p(buf286.data_ptr()), c_void_p(primals_509.data_ptr()), c_void_p(primals_510.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf287.data_ptr()))
    del primals_146
    # Source Nodes: [new_features_44], Original ATen: [aten.convolution]
    buf288 = extern_kernels.convolution(buf287, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf288, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf294 = reinterpret_tensor(buf295, (4, 32, 14, 14), (81536, 1, 5824, 416), 384)  # alias
    buf305 = reinterpret_tensor(buf307, (4, 32, 14, 14), (87808, 1, 6272, 448), 384)  # alias
    buf317 = reinterpret_tensor(buf320, (4, 32, 14, 14), (94080, 1, 6720, 480), 384)  # alias
    buf330 = reinterpret_tensor(buf334, (4, 32, 14, 14), (100352, 1, 7168, 512), 384)  # alias
    buf344 = reinterpret_tensor(buf349, (4, 32, 14, 14), (106624, 1, 7616, 544), 384)  # alias
    buf296 = empty_strided((4, 416, 14, 14), (81536, 1, 5824, 416), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_49(c_void_p(buf288.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(primals_512.data_ptr()), c_void_p(primals_513.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf296.data_ptr()))
    del primals_149
    # Source Nodes: [bottleneck_output_46], Original ATen: [aten.convolution]
    buf297 = extern_kernels.convolution(buf296, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf297, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf298 = reinterpret_tensor(buf144, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf144  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_50(c_void_p(buf297.data_ptr()), c_void_p(primals_515.data_ptr()), c_void_p(primals_516.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf298.data_ptr()))
    del primals_152
    # Source Nodes: [new_features_46], Original ATen: [aten.convolution]
    buf299 = extern_kernels.convolution(buf298, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf299, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf306 = reinterpret_tensor(buf307, (4, 32, 14, 14), (87808, 1, 6272, 448), 416)  # alias
    buf318 = reinterpret_tensor(buf320, (4, 32, 14, 14), (94080, 1, 6720, 480), 416)  # alias
    buf331 = reinterpret_tensor(buf334, (4, 32, 14, 14), (100352, 1, 7168, 512), 416)  # alias
    buf345 = reinterpret_tensor(buf349, (4, 32, 14, 14), (106624, 1, 7616, 544), 416)  # alias
    buf365 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    buf360 = reinterpret_tensor(buf365, (4, 32, 14, 14), (112896, 1, 8064, 576), 416)  # alias
    buf308 = empty_strided((4, 448, 14, 14), (87808, 1, 6272, 448), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_51(c_void_p(buf299.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(primals_518.data_ptr()), c_void_p(primals_519.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf308.data_ptr()))
    del primals_155
    # Source Nodes: [bottleneck_output_48], Original ATen: [aten.convolution]
    buf309 = extern_kernels.convolution(buf308, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf309, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf310 = reinterpret_tensor(buf134, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf134  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_52(c_void_p(buf309.data_ptr()), c_void_p(primals_521.data_ptr()), c_void_p(primals_522.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf310.data_ptr()))
    del primals_158
    # Source Nodes: [new_features_48], Original ATen: [aten.convolution]
    buf311 = extern_kernels.convolution(buf310, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf311, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf319 = reinterpret_tensor(buf320, (4, 32, 14, 14), (94080, 1, 6720, 480), 448)  # alias
    buf332 = reinterpret_tensor(buf334, (4, 32, 14, 14), (100352, 1, 7168, 512), 448)  # alias
    buf346 = reinterpret_tensor(buf349, (4, 32, 14, 14), (106624, 1, 7616, 544), 448)  # alias
    buf361 = reinterpret_tensor(buf365, (4, 32, 14, 14), (112896, 1, 8064, 576), 448)  # alias
    buf382 = empty_strided((4, 608, 14, 14), (119168, 1, 8512, 608), device='cpu', dtype=torch.float32)
    buf377 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 448)  # alias
    buf321 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_53(c_void_p(buf311.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(primals_524.data_ptr()), c_void_p(primals_525.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf321.data_ptr()))
    del primals_161
    # Source Nodes: [bottleneck_output_50], Original ATen: [aten.convolution]
    buf322 = extern_kernels.convolution(buf321, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf322, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf323 = reinterpret_tensor(buf129, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf129  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_54(c_void_p(buf322.data_ptr()), c_void_p(primals_527.data_ptr()), c_void_p(primals_528.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf323.data_ptr()))
    del primals_164
    # Source Nodes: [new_features_50], Original ATen: [aten.convolution]
    buf324 = extern_kernels.convolution(buf323, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf324, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf333 = reinterpret_tensor(buf334, (4, 32, 14, 14), (100352, 1, 7168, 512), 480)  # alias
    buf347 = reinterpret_tensor(buf349, (4, 32, 14, 14), (106624, 1, 7616, 544), 480)  # alias
    buf362 = reinterpret_tensor(buf365, (4, 32, 14, 14), (112896, 1, 8064, 576), 480)  # alias
    buf378 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 480)  # alias
    buf400 = empty_strided((4, 640, 14, 14), (125440, 1, 8960, 640), device='cpu', dtype=torch.float32)
    buf395 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 480)  # alias
    buf335 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_55(c_void_p(buf324.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(primals_530.data_ptr()), c_void_p(primals_531.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf335.data_ptr()))
    del primals_167
    # Source Nodes: [bottleneck_output_52], Original ATen: [aten.convolution]
    buf336 = extern_kernels.convolution(buf335, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf336, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf337 = reinterpret_tensor(buf124, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf124  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_56(c_void_p(buf336.data_ptr()), c_void_p(primals_533.data_ptr()), c_void_p(primals_534.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(buf337.data_ptr()))
    del primals_170
    # Source Nodes: [new_features_52], Original ATen: [aten.convolution]
    buf338 = extern_kernels.convolution(buf337, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf338, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf348 = reinterpret_tensor(buf349, (4, 32, 14, 14), (106624, 1, 7616, 544), 512)  # alias
    buf363 = reinterpret_tensor(buf365, (4, 32, 14, 14), (112896, 1, 8064, 576), 512)  # alias
    buf379 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 512)  # alias
    buf396 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 512)  # alias
    buf350 = empty_strided((4, 544, 14, 14), (106624, 1, 7616, 544), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_57(c_void_p(buf338.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(primals_536.data_ptr()), c_void_p(primals_537.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf350.data_ptr()))
    del primals_173
    # Source Nodes: [bottleneck_output_54], Original ATen: [aten.convolution]
    buf351 = extern_kernels.convolution(buf350, primals_174, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf351, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf352 = reinterpret_tensor(buf119, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf119  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_58(c_void_p(buf351.data_ptr()), c_void_p(primals_539.data_ptr()), c_void_p(primals_540.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(buf352.data_ptr()))
    del primals_176
    # Source Nodes: [new_features_54], Original ATen: [aten.convolution]
    buf353 = extern_kernels.convolution(buf352, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf353, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf354 = reinterpret_tensor(buf365, (4, 256, 14, 14), (112896, 1, 8064, 576), 0)  # alias
    buf370 = reinterpret_tensor(buf382, (4, 256, 14, 14), (119168, 1, 8512, 608), 0)  # alias
    buf387 = reinterpret_tensor(buf400, (4, 256, 14, 14), (125440, 1, 8960, 640), 0)  # alias
    buf419 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    buf405 = reinterpret_tensor(buf419, (4, 256, 14, 14), (131712, 1, 9408, 672), 0)  # alias
    buf355 = reinterpret_tensor(buf365, (4, 32, 14, 14), (112896, 1, 8064, 576), 256)  # alias
    buf371 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 256)  # alias
    buf388 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 256)  # alias
    buf406 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 256)  # alias
    buf356 = reinterpret_tensor(buf365, (4, 32, 14, 14), (112896, 1, 8064, 576), 288)  # alias
    buf372 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 288)  # alias
    buf389 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 288)  # alias
    buf407 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 288)  # alias
    buf357 = reinterpret_tensor(buf365, (4, 32, 14, 14), (112896, 1, 8064, 576), 320)  # alias
    buf373 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 320)  # alias
    buf390 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 320)  # alias
    buf408 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 320)  # alias
    buf358 = reinterpret_tensor(buf365, (4, 32, 14, 14), (112896, 1, 8064, 576), 352)  # alias
    buf374 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 352)  # alias
    buf391 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 352)  # alias
    buf409 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 352)  # alias
    buf359 = reinterpret_tensor(buf365, (4, 32, 14, 14), (112896, 1, 8064, 576), 384)  # alias
    buf375 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 384)  # alias
    buf392 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 384)  # alias
    buf410 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 384)  # alias
    buf364 = reinterpret_tensor(buf365, (4, 32, 14, 14), (112896, 1, 8064, 576), 544)  # alias
    buf380 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 544)  # alias
    buf397 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 544)  # alias
    buf415 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 544)  # alias
    buf366 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_59(c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(primals_542.data_ptr()), c_void_p(primals_543.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf366.data_ptr()))
    del primals_179
    # Source Nodes: [bottleneck_output_56], Original ATen: [aten.convolution]
    buf367 = extern_kernels.convolution(buf366, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf367, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf368 = reinterpret_tensor(buf194, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf194  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_60(c_void_p(buf367.data_ptr()), c_void_p(primals_545.data_ptr()), c_void_p(primals_546.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(buf368.data_ptr()))
    del primals_182
    # Source Nodes: [new_features_56], Original ATen: [aten.convolution]
    buf369 = extern_kernels.convolution(buf368, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf369, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf376 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 416)  # alias
    buf393 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 416)  # alias
    buf411 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 416)  # alias
    buf439 = empty_strided((4, 704, 14, 14), (137984, 1, 9856, 704), device='cpu', dtype=torch.float32)
    buf430 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 416)  # alias
    buf381 = reinterpret_tensor(buf382, (4, 32, 14, 14), (119168, 1, 8512, 608), 576)  # alias
    buf398 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 576)  # alias
    buf416 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 576)  # alias
    buf435 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 576)  # alias
    buf383 = empty_strided((4, 608, 14, 14), (119168, 1, 8512, 608), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_61(c_void_p(buf299.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(primals_548.data_ptr()), c_void_p(primals_549.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf383.data_ptr()))
    del primals_185
    # Source Nodes: [bottleneck_output_58], Original ATen: [aten.convolution]
    buf384 = extern_kernels.convolution(buf383, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf384, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf385 = reinterpret_tensor(buf180, (4, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf180  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_62(c_void_p(buf384.data_ptr()), c_void_p(primals_551.data_ptr()), c_void_p(primals_552.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(buf385.data_ptr()))
    del primals_188
    # Source Nodes: [new_features_58], Original ATen: [aten.convolution]
    buf386 = extern_kernels.convolution(buf385, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf386, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf394 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 448)  # alias
    buf412 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 448)  # alias
    buf431 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 448)  # alias
    buf460 = empty_strided((4, 736, 14, 14), (144256, 1, 10304, 736), device='cpu', dtype=torch.float32)
    buf451 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 448)  # alias
    buf399 = reinterpret_tensor(buf400, (4, 32, 14, 14), (125440, 1, 8960, 640), 608)  # alias
    buf417 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 608)  # alias
    buf436 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 608)  # alias
    buf456 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 608)  # alias
    buf401 = empty_strided((4, 640, 14, 14), (125440, 1, 8960, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_63(c_void_p(buf311.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(primals_554.data_ptr()), c_void_p(primals_555.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf401.data_ptr()))
    del primals_191
    # Source Nodes: [bottleneck_output_60], Original ATen: [aten.convolution]
    buf402 = extern_kernels.convolution(buf401, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf402, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf403 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_64(c_void_p(buf402.data_ptr()), c_void_p(primals_557.data_ptr()), c_void_p(primals_558.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(buf403.data_ptr()))
    del primals_194
    # Source Nodes: [new_features_60], Original ATen: [aten.convolution]
    buf404 = extern_kernels.convolution(buf403, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf404, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf413 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 480)  # alias
    buf432 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 480)  # alias
    buf452 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 480)  # alias
    buf482 = empty_strided((4, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    buf473 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 480)  # alias
    buf414 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 512)  # alias
    buf433 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 512)  # alias
    buf453 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 512)  # alias
    buf474 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 512)  # alias
    buf418 = reinterpret_tensor(buf419, (4, 32, 14, 14), (131712, 1, 9408, 672), 640)  # alias
    buf437 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 640)  # alias
    buf457 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 640)  # alias
    buf478 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 640)  # alias
    buf420 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_65(c_void_p(buf324.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(primals_560.data_ptr()), c_void_p(primals_561.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf420.data_ptr()))
    del primals_197
    # Source Nodes: [bottleneck_output_62], Original ATen: [aten.convolution]
    buf421 = extern_kernels.convolution(buf420, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf421, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf422 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_66(c_void_p(buf421.data_ptr()), c_void_p(primals_563.data_ptr()), c_void_p(primals_564.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(buf422.data_ptr()))
    del primals_200
    # Source Nodes: [new_features_62], Original ATen: [aten.convolution]
    buf423 = extern_kernels.convolution(buf422, buf32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf423, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf424 = reinterpret_tensor(buf439, (4, 256, 14, 14), (137984, 1, 9856, 704), 0)  # alias
    buf444 = reinterpret_tensor(buf460, (4, 256, 14, 14), (144256, 1, 10304, 736), 0)  # alias
    buf465 = reinterpret_tensor(buf482, (4, 256, 14, 14), (150528, 1, 10752, 768), 0)  # alias
    buf505 = empty_strided((4, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    buf487 = reinterpret_tensor(buf505, (4, 256, 14, 14), (156800, 1, 11200, 800), 0)  # alias
    buf425 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 256)  # alias
    buf445 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 256)  # alias
    buf466 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 256)  # alias
    buf488 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 256)  # alias
    buf426 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 288)  # alias
    buf446 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 288)  # alias
    buf467 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 288)  # alias
    buf489 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 288)  # alias
    buf427 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 320)  # alias
    buf447 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 320)  # alias
    buf468 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 320)  # alias
    buf490 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 320)  # alias
    buf428 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 352)  # alias
    buf448 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 352)  # alias
    buf469 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 352)  # alias
    buf491 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 352)  # alias
    buf429 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 384)  # alias
    buf449 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 384)  # alias
    buf470 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 384)  # alias
    buf492 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 384)  # alias
    buf434 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 544)  # alias
    buf454 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 544)  # alias
    buf475 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 544)  # alias
    buf497 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 544)  # alias
    buf438 = reinterpret_tensor(buf439, (4, 32, 14, 14), (137984, 1, 9856, 704), 672)  # alias
    buf458 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 672)  # alias
    buf479 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 672)  # alias
    buf501 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 672)  # alias
    buf440 = empty_strided((4, 704, 14, 14), (137984, 1, 9856, 704), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_67(c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(primals_566.data_ptr()), c_void_p(primals_567.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf440.data_ptr()))
    del primals_203
    # Source Nodes: [bottleneck_output_64], Original ATen: [aten.convolution]
    buf441 = extern_kernels.convolution(buf440, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf441, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf442 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_68(c_void_p(buf441.data_ptr()), c_void_p(primals_569.data_ptr()), c_void_p(primals_570.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf442.data_ptr()))
    del primals_206
    # Source Nodes: [new_features_64], Original ATen: [aten.convolution]
    buf443 = extern_kernels.convolution(buf442, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf443, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf450 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 416)  # alias
    buf471 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 416)  # alias
    buf493 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 416)  # alias
    buf455 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 576)  # alias
    buf476 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 576)  # alias
    buf498 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 576)  # alias
    buf459 = reinterpret_tensor(buf460, (4, 32, 14, 14), (144256, 1, 10304, 736), 704)  # alias
    buf480 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 704)  # alias
    buf502 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 704)  # alias
    buf461 = empty_strided((4, 736, 14, 14), (144256, 1, 10304, 736), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_69(c_void_p(buf299.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(primals_572.data_ptr()), c_void_p(primals_573.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf461.data_ptr()))
    del primals_209
    # Source Nodes: [bottleneck_output_66], Original ATen: [aten.convolution]
    buf462 = extern_kernels.convolution(buf461, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf462, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf463 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_70(c_void_p(buf462.data_ptr()), c_void_p(primals_575.data_ptr()), c_void_p(primals_576.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf463.data_ptr()))
    del primals_212
    # Source Nodes: [new_features_66], Original ATen: [aten.convolution]
    buf464 = extern_kernels.convolution(buf463, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf464, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf472 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 448)  # alias
    buf494 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 448)  # alias
    buf529 = empty_strided((4, 832, 14, 14), (163072, 1, 11648, 832), device='cpu', dtype=torch.float32)
    buf517 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 448)  # alias
    buf477 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 608)  # alias
    buf499 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 608)  # alias
    buf522 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 608)  # alias
    buf481 = reinterpret_tensor(buf482, (4, 32, 14, 14), (150528, 1, 10752, 768), 736)  # alias
    buf503 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 736)  # alias
    buf526 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 736)  # alias
    buf483 = empty_strided((4, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_71(c_void_p(buf311.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(primals_578.data_ptr()), c_void_p(primals_579.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf483.data_ptr()))
    del primals_215
    # Source Nodes: [bottleneck_output_68], Original ATen: [aten.convolution]
    buf484 = extern_kernels.convolution(buf483, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf484, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf485 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_72(c_void_p(buf484.data_ptr()), c_void_p(primals_581.data_ptr()), c_void_p(primals_582.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(buf485.data_ptr()))
    del primals_218
    # Source Nodes: [new_features_68], Original ATen: [aten.convolution]
    buf486 = extern_kernels.convolution(buf485, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf486, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf495 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 480)  # alias
    buf518 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 480)  # alias
    buf554 = empty_strided((4, 864, 14, 14), (169344, 1, 12096, 864), device='cpu', dtype=torch.float32)
    buf542 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 480)  # alias
    buf496 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 512)  # alias
    buf519 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 512)  # alias
    buf543 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 512)  # alias
    buf500 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 640)  # alias
    buf523 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 640)  # alias
    buf547 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 640)  # alias
    buf504 = reinterpret_tensor(buf505, (4, 32, 14, 14), (156800, 1, 11200, 800), 768)  # alias
    buf527 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 768)  # alias
    buf551 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 768)  # alias
    buf506 = empty_strided((4, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_73(c_void_p(buf324.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(primals_584.data_ptr()), c_void_p(primals_585.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf506.data_ptr()))
    del primals_221
    # Source Nodes: [bottleneck_output_70], Original ATen: [aten.convolution]
    buf507 = extern_kernels.convolution(buf506, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf507, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf508 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_74(c_void_p(buf507.data_ptr()), c_void_p(primals_587.data_ptr()), c_void_p(primals_588.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(buf508.data_ptr()))
    del primals_224
    # Source Nodes: [new_features_70], Original ATen: [aten.convolution]
    buf509 = extern_kernels.convolution(buf508, buf36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf509, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf510 = reinterpret_tensor(buf529, (4, 256, 14, 14), (163072, 1, 11648, 832), 0)  # alias
    buf534 = reinterpret_tensor(buf554, (4, 256, 14, 14), (169344, 1, 12096, 864), 0)  # alias
    buf580 = empty_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    buf559 = reinterpret_tensor(buf580, (4, 256, 14, 14), (175616, 1, 12544, 896), 0)  # alias
    buf511 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 256)  # alias
    buf535 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 256)  # alias
    buf560 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 256)  # alias
    buf512 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 288)  # alias
    buf536 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 288)  # alias
    buf561 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 288)  # alias
    buf513 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 320)  # alias
    buf537 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 320)  # alias
    buf562 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 320)  # alias
    buf514 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 352)  # alias
    buf538 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 352)  # alias
    buf563 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 352)  # alias
    buf515 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 384)  # alias
    buf539 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 384)  # alias
    buf564 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 384)  # alias
    buf516 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 416)  # alias
    buf540 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 416)  # alias
    buf565 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 416)  # alias
    buf520 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 544)  # alias
    buf544 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 544)  # alias
    buf569 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 544)  # alias
    buf521 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 576)  # alias
    buf545 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 576)  # alias
    buf570 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 576)  # alias
    buf524 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 672)  # alias
    buf548 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 672)  # alias
    buf573 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 672)  # alias
    buf525 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 704)  # alias
    buf549 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 704)  # alias
    buf574 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 704)  # alias
    buf528 = reinterpret_tensor(buf529, (4, 32, 14, 14), (163072, 1, 11648, 832), 800)  # alias
    buf552 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 800)  # alias
    buf577 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 800)  # alias
    buf530 = empty_strided((4, 832, 14, 14), (163072, 1, 11648, 832), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_75(c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(primals_590.data_ptr()), c_void_p(primals_591.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf530.data_ptr()))
    del primals_227
    # Source Nodes: [bottleneck_output_72], Original ATen: [aten.convolution]
    buf531 = extern_kernels.convolution(buf530, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf531, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf532 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_76(c_void_p(buf531.data_ptr()), c_void_p(primals_593.data_ptr()), c_void_p(primals_594.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(buf532.data_ptr()))
    del primals_230
    # Source Nodes: [new_features_72], Original ATen: [aten.convolution]
    buf533 = extern_kernels.convolution(buf532, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf533, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf541 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 448)  # alias
    buf566 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 448)  # alias
    buf607 = empty_strided((4, 928, 14, 14), (181888, 1, 12992, 928), device='cpu', dtype=torch.float32)
    buf592 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 448)  # alias
    buf546 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 608)  # alias
    buf571 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 608)  # alias
    buf597 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 608)  # alias
    buf550 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 736)  # alias
    buf575 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 736)  # alias
    buf601 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 736)  # alias
    buf553 = reinterpret_tensor(buf554, (4, 32, 14, 14), (169344, 1, 12096, 864), 832)  # alias
    buf578 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 832)  # alias
    buf604 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 832)  # alias
    buf555 = empty_strided((4, 864, 14, 14), (169344, 1, 12096, 864), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_77(c_void_p(buf311.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(primals_596.data_ptr()), c_void_p(primals_597.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf555.data_ptr()))
    del primals_233
    # Source Nodes: [bottleneck_output_74], Original ATen: [aten.convolution]
    buf556 = extern_kernels.convolution(buf555, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf556, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf557 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_78(c_void_p(buf556.data_ptr()), c_void_p(primals_599.data_ptr()), c_void_p(primals_600.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(buf557.data_ptr()))
    del primals_236
    # Source Nodes: [new_features_74], Original ATen: [aten.convolution]
    buf558 = extern_kernels.convolution(buf557, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf558, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf567 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 480)  # alias
    buf593 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 480)  # alias
    buf635 = empty_strided((4, 960, 14, 14), (188160, 1, 13440, 960), device='cpu', dtype=torch.float32)
    buf620 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 480)  # alias
    buf568 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 512)  # alias
    buf594 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 512)  # alias
    buf621 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 512)  # alias
    buf572 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 640)  # alias
    buf598 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 640)  # alias
    buf625 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 640)  # alias
    buf576 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 768)  # alias
    buf602 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 768)  # alias
    buf629 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 768)  # alias
    buf579 = reinterpret_tensor(buf580, (4, 32, 14, 14), (175616, 1, 12544, 896), 864)  # alias
    buf605 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 864)  # alias
    buf632 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 864)  # alias
    buf581 = empty_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_79(c_void_p(buf324.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(primals_602.data_ptr()), c_void_p(primals_603.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf581.data_ptr()))
    del primals_239
    # Source Nodes: [bottleneck_output_76], Original ATen: [aten.convolution]
    buf582 = extern_kernels.convolution(buf581, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf582, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf583 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_80(c_void_p(buf582.data_ptr()), c_void_p(primals_605.data_ptr()), c_void_p(primals_606.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(buf583.data_ptr()))
    del primals_242
    # Source Nodes: [new_features_76], Original ATen: [aten.convolution]
    buf584 = extern_kernels.convolution(buf583, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf584, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf585 = reinterpret_tensor(buf607, (4, 256, 14, 14), (181888, 1, 12992, 928), 0)  # alias
    buf612 = reinterpret_tensor(buf635, (4, 256, 14, 14), (188160, 1, 13440, 960), 0)  # alias
    buf664 = empty_strided((4, 992, 14, 14), (194432, 1, 13888, 992), device='cpu', dtype=torch.float32)
    buf640 = reinterpret_tensor(buf664, (4, 256, 14, 14), (194432, 1, 13888, 992), 0)  # alias
    buf586 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 256)  # alias
    buf613 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 256)  # alias
    buf641 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 256)  # alias
    buf587 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 288)  # alias
    buf614 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 288)  # alias
    buf642 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 288)  # alias
    buf588 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 320)  # alias
    buf615 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 320)  # alias
    buf643 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 320)  # alias
    buf589 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 352)  # alias
    buf616 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 352)  # alias
    buf644 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 352)  # alias
    buf590 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 384)  # alias
    buf617 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 384)  # alias
    buf645 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 384)  # alias
    buf591 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 416)  # alias
    buf618 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 416)  # alias
    buf646 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 416)  # alias
    buf595 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 544)  # alias
    buf622 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 544)  # alias
    buf650 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 544)  # alias
    buf596 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 576)  # alias
    buf623 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 576)  # alias
    buf651 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 576)  # alias
    buf599 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 672)  # alias
    buf626 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 672)  # alias
    buf654 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 672)  # alias
    buf600 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 704)  # alias
    buf627 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 704)  # alias
    buf655 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 704)  # alias
    buf603 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 800)  # alias
    buf630 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 800)  # alias
    buf658 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 800)  # alias
    buf606 = reinterpret_tensor(buf607, (4, 32, 14, 14), (181888, 1, 12992, 928), 896)  # alias
    buf633 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 896)  # alias
    buf661 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 896)  # alias
    buf608 = empty_strided((4, 928, 14, 14), (181888, 1, 12992, 928), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_81(c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(primals_608.data_ptr()), c_void_p(primals_609.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf643.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf608.data_ptr()))
    del primals_245
    # Source Nodes: [bottleneck_output_78], Original ATen: [aten.convolution]
    buf609 = extern_kernels.convolution(buf608, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf609, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf610 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_82(c_void_p(buf609.data_ptr()), c_void_p(primals_611.data_ptr()), c_void_p(primals_612.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(buf610.data_ptr()))
    del primals_248
    # Source Nodes: [new_features_78], Original ATen: [aten.convolution]
    buf611 = extern_kernels.convolution(buf610, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf611, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf619 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 448)  # alias
    buf647 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 448)  # alias
    buf694 = empty_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    buf676 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 448)  # alias
    buf624 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 608)  # alias
    buf652 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 608)  # alias
    buf681 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 608)  # alias
    buf628 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 736)  # alias
    buf656 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 736)  # alias
    buf685 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 736)  # alias
    buf631 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 832)  # alias
    buf659 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 832)  # alias
    buf688 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 832)  # alias
    buf634 = reinterpret_tensor(buf635, (4, 32, 14, 14), (188160, 1, 13440, 960), 928)  # alias
    buf662 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 928)  # alias
    buf691 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 928)  # alias
    buf636 = empty_strided((4, 960, 14, 14), (188160, 1, 13440, 960), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_83(c_void_p(buf311.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(primals_614.data_ptr()), c_void_p(primals_615.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf676.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf636.data_ptr()))
    del buf311
    del buf386
    del buf464
    del buf533
    del buf611
    del primals_251
    # Source Nodes: [bottleneck_output_80], Original ATen: [aten.convolution]
    buf637 = extern_kernels.convolution(buf636, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf637, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf638 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_84(c_void_p(buf637.data_ptr()), c_void_p(primals_617.data_ptr()), c_void_p(primals_618.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(buf638.data_ptr()))
    del primals_254
    # Source Nodes: [new_features_80], Original ATen: [aten.convolution]
    buf639 = extern_kernels.convolution(buf638, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf639, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf648 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 480)  # alias
    buf677 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 480)  # alias
    buf649 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 512)  # alias
    buf678 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
    buf653 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 640)  # alias
    buf682 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 640)  # alias
    buf657 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 768)  # alias
    buf686 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 768)  # alias
    buf660 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 864)  # alias
    buf689 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 864)  # alias
    buf663 = reinterpret_tensor(buf664, (4, 32, 14, 14), (194432, 1, 13888, 992), 960)  # alias
    buf692 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 960)  # alias
    buf665 = empty_strided((4, 992, 14, 14), (194432, 1, 13888, 992), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_85(c_void_p(buf324.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(primals_620.data_ptr()), c_void_p(primals_621.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf692.data_ptr()), c_void_p(buf665.data_ptr()))
    del buf324
    del buf338
    del buf404
    del primals_257
    # Source Nodes: [bottleneck_output_82], Original ATen: [aten.convolution]
    buf666 = extern_kernels.convolution(buf665, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf666, (4, 128, 14, 14), (25088, 1, 1792, 128))
    buf667 = empty_strided((4, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_86(c_void_p(buf666.data_ptr()), c_void_p(primals_623.data_ptr()), c_void_p(primals_624.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(buf667.data_ptr()))
    del primals_260
    # Source Nodes: [new_features_82], Original ATen: [aten.convolution]
    buf668 = extern_kernels.convolution(buf667, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf668, (4, 32, 14, 14), (6272, 1, 448, 32))
    buf669 = reinterpret_tensor(buf694, (4, 256, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
    buf670 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 256)  # alias
    buf671 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 288)  # alias
    buf672 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 320)  # alias
    buf673 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 352)  # alias
    buf674 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 384)  # alias
    buf675 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 416)  # alias
    buf679 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 544)  # alias
    buf680 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 576)  # alias
    buf683 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 672)  # alias
    buf684 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 704)  # alias
    buf687 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 800)  # alias
    buf690 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 896)  # alias
    buf693 = reinterpret_tensor(buf694, (4, 32, 14, 14), (200704, 1, 14336, 1024), 992)  # alias
    buf695 = empty_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_87(c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(primals_626.data_ptr()), c_void_p(primals_627.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf684.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf695.data_ptr()))
    del primals_263
    # Source Nodes: [l__mod___features_transition3_conv], Original ATen: [aten.convolution]
    buf696 = extern_kernels.convolution(buf695, primals_264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf696, (4, 512, 14, 14), (100352, 1, 7168, 512))
    buf697 = empty_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    buf698 = empty_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_88(c_void_p(buf696.data_ptr()), c_void_p(primals_629.data_ptr()), c_void_p(primals_630.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf698.data_ptr()))
    del primals_266
    # Source Nodes: [bottleneck_output_84], Original ATen: [aten.convolution]
    buf699 = extern_kernels.convolution(buf698, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf699, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf700 = reinterpret_tensor(buf668, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf668  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_89(c_void_p(buf699.data_ptr()), c_void_p(primals_632.data_ptr()), c_void_p(primals_633.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(buf700.data_ptr()))
    del primals_269
    # Source Nodes: [new_features_84], Original ATen: [aten.convolution]
    buf701 = extern_kernels.convolution(buf700, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf701, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf702 = empty_strided((4, 544, 7, 7), (26656, 1, 3808, 544), device='cpu', dtype=torch.float32)
    buf703 = empty_strided((4, 544, 7, 7), (26656, 1, 3808, 544), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_90(c_void_p(buf697.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(primals_635.data_ptr()), c_void_p(primals_636.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf703.data_ptr()))
    del primals_272
    # Source Nodes: [bottleneck_output_86], Original ATen: [aten.convolution]
    buf704 = extern_kernels.convolution(buf703, primals_273, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf704, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf705 = reinterpret_tensor(buf584, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf584  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_91(c_void_p(buf704.data_ptr()), c_void_p(primals_638.data_ptr()), c_void_p(primals_639.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(buf705.data_ptr()))
    del primals_275
    # Source Nodes: [new_features_86], Original ATen: [aten.convolution]
    buf706 = extern_kernels.convolution(buf705, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf706, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf707 = empty_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cpu', dtype=torch.float32)
    buf708 = empty_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_92(c_void_p(buf697.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(primals_641.data_ptr()), c_void_p(primals_642.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf708.data_ptr()))
    del primals_278
    # Source Nodes: [bottleneck_output_88], Original ATen: [aten.convolution]
    buf709 = extern_kernels.convolution(buf708, primals_279, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf709, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf710 = reinterpret_tensor(buf509, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf509  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_93(c_void_p(buf709.data_ptr()), c_void_p(primals_644.data_ptr()), c_void_p(primals_645.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(buf710.data_ptr()))
    del primals_281
    # Source Nodes: [new_features_88], Original ATen: [aten.convolution]
    buf711 = extern_kernels.convolution(buf710, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf711, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf712 = empty_strided((4, 608, 7, 7), (29792, 1, 4256, 608), device='cpu', dtype=torch.float32)
    buf713 = empty_strided((4, 608, 7, 7), (29792, 1, 4256, 608), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_94(c_void_p(buf697.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(primals_647.data_ptr()), c_void_p(primals_648.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf713.data_ptr()))
    del primals_284
    # Source Nodes: [bottleneck_output_90], Original ATen: [aten.convolution]
    buf714 = extern_kernels.convolution(buf713, primals_285, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf714, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf715 = reinterpret_tensor(buf443, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf443  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_95(c_void_p(buf714.data_ptr()), c_void_p(primals_650.data_ptr()), c_void_p(primals_651.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(buf715.data_ptr()))
    del primals_287
    # Source Nodes: [new_features_90], Original ATen: [aten.convolution]
    buf716 = extern_kernels.convolution(buf715, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf716, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf722 = empty_strided((4, 640, 7, 7), (31360, 1, 4480, 640), device='cpu', dtype=torch.float32)
    buf717 = reinterpret_tensor(buf722, (4, 512, 7, 7), (31360, 1, 4480, 640), 0)  # alias
    buf733 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    buf727 = reinterpret_tensor(buf733, (4, 512, 7, 7), (32928, 1, 4704, 672), 0)  # alias
    buf745 = empty_strided((4, 704, 7, 7), (34496, 1, 4928, 704), device='cpu', dtype=torch.float32)
    buf738 = reinterpret_tensor(buf745, (4, 512, 7, 7), (34496, 1, 4928, 704), 0)  # alias
    buf758 = empty_strided((4, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    buf750 = reinterpret_tensor(buf758, (4, 512, 7, 7), (36064, 1, 5152, 736), 0)  # alias
    buf772 = empty_strided((4, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    buf763 = reinterpret_tensor(buf772, (4, 512, 7, 7), (37632, 1, 5376, 768), 0)  # alias
    buf787 = empty_strided((4, 800, 7, 7), (39200, 1, 5600, 800), device='cpu', dtype=torch.float32)
    buf777 = reinterpret_tensor(buf787, (4, 512, 7, 7), (39200, 1, 5600, 800), 0)  # alias
    buf718 = reinterpret_tensor(buf722, (4, 32, 7, 7), (31360, 1, 4480, 640), 512)  # alias
    buf728 = reinterpret_tensor(buf733, (4, 32, 7, 7), (32928, 1, 4704, 672), 512)  # alias
    buf739 = reinterpret_tensor(buf745, (4, 32, 7, 7), (34496, 1, 4928, 704), 512)  # alias
    buf751 = reinterpret_tensor(buf758, (4, 32, 7, 7), (36064, 1, 5152, 736), 512)  # alias
    buf764 = reinterpret_tensor(buf772, (4, 32, 7, 7), (37632, 1, 5376, 768), 512)  # alias
    buf778 = reinterpret_tensor(buf787, (4, 32, 7, 7), (39200, 1, 5600, 800), 512)  # alias
    buf719 = reinterpret_tensor(buf722, (4, 32, 7, 7), (31360, 1, 4480, 640), 544)  # alias
    buf729 = reinterpret_tensor(buf733, (4, 32, 7, 7), (32928, 1, 4704, 672), 544)  # alias
    buf740 = reinterpret_tensor(buf745, (4, 32, 7, 7), (34496, 1, 4928, 704), 544)  # alias
    buf752 = reinterpret_tensor(buf758, (4, 32, 7, 7), (36064, 1, 5152, 736), 544)  # alias
    buf765 = reinterpret_tensor(buf772, (4, 32, 7, 7), (37632, 1, 5376, 768), 544)  # alias
    buf779 = reinterpret_tensor(buf787, (4, 32, 7, 7), (39200, 1, 5600, 800), 544)  # alias
    buf720 = reinterpret_tensor(buf722, (4, 32, 7, 7), (31360, 1, 4480, 640), 576)  # alias
    buf730 = reinterpret_tensor(buf733, (4, 32, 7, 7), (32928, 1, 4704, 672), 576)  # alias
    buf741 = reinterpret_tensor(buf745, (4, 32, 7, 7), (34496, 1, 4928, 704), 576)  # alias
    buf753 = reinterpret_tensor(buf758, (4, 32, 7, 7), (36064, 1, 5152, 736), 576)  # alias
    buf766 = reinterpret_tensor(buf772, (4, 32, 7, 7), (37632, 1, 5376, 768), 576)  # alias
    buf780 = reinterpret_tensor(buf787, (4, 32, 7, 7), (39200, 1, 5600, 800), 576)  # alias
    buf721 = reinterpret_tensor(buf722, (4, 32, 7, 7), (31360, 1, 4480, 640), 608)  # alias
    buf731 = reinterpret_tensor(buf733, (4, 32, 7, 7), (32928, 1, 4704, 672), 608)  # alias
    buf742 = reinterpret_tensor(buf745, (4, 32, 7, 7), (34496, 1, 4928, 704), 608)  # alias
    buf754 = reinterpret_tensor(buf758, (4, 32, 7, 7), (36064, 1, 5152, 736), 608)  # alias
    buf767 = reinterpret_tensor(buf772, (4, 32, 7, 7), (37632, 1, 5376, 768), 608)  # alias
    buf781 = reinterpret_tensor(buf787, (4, 32, 7, 7), (39200, 1, 5600, 800), 608)  # alias
    buf723 = empty_strided((4, 640, 7, 7), (31360, 1, 4480, 640), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_96(c_void_p(buf697.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(primals_653.data_ptr()), c_void_p(primals_654.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(buf738.data_ptr()), c_void_p(buf750.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf751.data_ptr()), c_void_p(buf764.data_ptr()), c_void_p(buf778.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf752.data_ptr()), c_void_p(buf765.data_ptr()), c_void_p(buf779.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf730.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(buf766.data_ptr()), c_void_p(buf780.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(buf781.data_ptr()), c_void_p(buf723.data_ptr()))
    del primals_290
    # Source Nodes: [bottleneck_output_92], Original ATen: [aten.convolution]
    buf724 = extern_kernels.convolution(buf723, primals_291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf724, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf725 = reinterpret_tensor(buf423, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf423  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_97(c_void_p(buf724.data_ptr()), c_void_p(primals_656.data_ptr()), c_void_p(primals_657.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(buf725.data_ptr()))
    del primals_293
    # Source Nodes: [new_features_92], Original ATen: [aten.convolution]
    buf726 = extern_kernels.convolution(buf725, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf726, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf732 = reinterpret_tensor(buf733, (4, 32, 7, 7), (32928, 1, 4704, 672), 640)  # alias
    buf743 = reinterpret_tensor(buf745, (4, 32, 7, 7), (34496, 1, 4928, 704), 640)  # alias
    buf755 = reinterpret_tensor(buf758, (4, 32, 7, 7), (36064, 1, 5152, 736), 640)  # alias
    buf768 = reinterpret_tensor(buf772, (4, 32, 7, 7), (37632, 1, 5376, 768), 640)  # alias
    buf782 = reinterpret_tensor(buf787, (4, 32, 7, 7), (39200, 1, 5600, 800), 640)  # alias
    buf734 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_98(c_void_p(buf726.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(primals_659.data_ptr()), c_void_p(primals_660.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(buf732.data_ptr()), c_void_p(buf743.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf768.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(buf734.data_ptr()))
    del primals_296
    # Source Nodes: [bottleneck_output_94], Original ATen: [aten.convolution]
    buf735 = extern_kernels.convolution(buf734, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf735, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf736 = reinterpret_tensor(buf369, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf369  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_99(c_void_p(buf735.data_ptr()), c_void_p(primals_662.data_ptr()), c_void_p(primals_663.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(buf736.data_ptr()))
    del primals_299
    # Source Nodes: [new_features_94], Original ATen: [aten.convolution]
    buf737 = extern_kernels.convolution(buf736, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf737, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf744 = reinterpret_tensor(buf745, (4, 32, 7, 7), (34496, 1, 4928, 704), 672)  # alias
    buf756 = reinterpret_tensor(buf758, (4, 32, 7, 7), (36064, 1, 5152, 736), 672)  # alias
    buf769 = reinterpret_tensor(buf772, (4, 32, 7, 7), (37632, 1, 5376, 768), 672)  # alias
    buf783 = reinterpret_tensor(buf787, (4, 32, 7, 7), (39200, 1, 5600, 800), 672)  # alias
    buf803 = empty_strided((4, 832, 7, 7), (40768, 1, 5824, 832), device='cpu', dtype=torch.float32)
    buf798 = reinterpret_tensor(buf803, (4, 32, 7, 7), (40768, 1, 5824, 832), 672)  # alias
    buf746 = empty_strided((4, 704, 7, 7), (34496, 1, 4928, 704), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_100(c_void_p(buf737.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(primals_665.data_ptr()), c_void_p(primals_666.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf756.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf798.data_ptr()), c_void_p(buf746.data_ptr()))
    del primals_302
    # Source Nodes: [bottleneck_output_96], Original ATen: [aten.convolution]
    buf747 = extern_kernels.convolution(buf746, primals_303, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf747, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf748 = reinterpret_tensor(buf353, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf353  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_101(c_void_p(buf747.data_ptr()), c_void_p(primals_668.data_ptr()), c_void_p(primals_669.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(buf748.data_ptr()))
    del primals_305
    # Source Nodes: [new_features_96], Original ATen: [aten.convolution]
    buf749 = extern_kernels.convolution(buf748, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf749, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf757 = reinterpret_tensor(buf758, (4, 32, 7, 7), (36064, 1, 5152, 736), 704)  # alias
    buf770 = reinterpret_tensor(buf772, (4, 32, 7, 7), (37632, 1, 5376, 768), 704)  # alias
    buf784 = reinterpret_tensor(buf787, (4, 32, 7, 7), (39200, 1, 5600, 800), 704)  # alias
    buf799 = reinterpret_tensor(buf803, (4, 32, 7, 7), (40768, 1, 5824, 832), 704)  # alias
    buf820 = empty_strided((4, 864, 7, 7), (42336, 1, 6048, 864), device='cpu', dtype=torch.float32)
    buf815 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 704)  # alias
    buf759 = empty_strided((4, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_102(c_void_p(buf749.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(primals_671.data_ptr()), c_void_p(primals_672.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf799.data_ptr()), c_void_p(buf815.data_ptr()), c_void_p(buf759.data_ptr()))
    del primals_308
    # Source Nodes: [bottleneck_output_98], Original ATen: [aten.convolution]
    buf760 = extern_kernels.convolution(buf759, primals_309, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf760, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf761 = reinterpret_tensor(buf299, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf299  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_103(c_void_p(buf760.data_ptr()), c_void_p(primals_674.data_ptr()), c_void_p(primals_675.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(buf761.data_ptr()))
    del primals_311
    # Source Nodes: [new_features_98], Original ATen: [aten.convolution]
    buf762 = extern_kernels.convolution(buf761, buf50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf762, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf771 = reinterpret_tensor(buf772, (4, 32, 7, 7), (37632, 1, 5376, 768), 736)  # alias
    buf785 = reinterpret_tensor(buf787, (4, 32, 7, 7), (39200, 1, 5600, 800), 736)  # alias
    buf800 = reinterpret_tensor(buf803, (4, 32, 7, 7), (40768, 1, 5824, 832), 736)  # alias
    buf816 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 736)  # alias
    buf838 = empty_strided((4, 896, 7, 7), (43904, 1, 6272, 896), device='cpu', dtype=torch.float32)
    buf833 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 736)  # alias
    buf773 = empty_strided((4, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_104(c_void_p(buf762.data_ptr()), c_void_p(buf772.data_ptr()), c_void_p(primals_677.data_ptr()), c_void_p(primals_678.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(buf771.data_ptr()), c_void_p(buf785.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf816.data_ptr()), c_void_p(buf833.data_ptr()), c_void_p(buf773.data_ptr()))
    del primals_314
    # Source Nodes: [bottleneck_output_100], Original ATen: [aten.convolution]
    buf774 = extern_kernels.convolution(buf773, primals_315, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf774, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf775 = reinterpret_tensor(buf288, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf288  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_105(c_void_p(buf774.data_ptr()), c_void_p(primals_680.data_ptr()), c_void_p(primals_681.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(buf775.data_ptr()))
    del primals_317
    # Source Nodes: [new_features_100], Original ATen: [aten.convolution]
    buf776 = extern_kernels.convolution(buf775, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf776, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf786 = reinterpret_tensor(buf787, (4, 32, 7, 7), (39200, 1, 5600, 800), 768)  # alias
    buf801 = reinterpret_tensor(buf803, (4, 32, 7, 7), (40768, 1, 5824, 832), 768)  # alias
    buf817 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 768)  # alias
    buf834 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 768)  # alias
    buf788 = empty_strided((4, 800, 7, 7), (39200, 1, 5600, 800), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_106(c_void_p(buf776.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(primals_683.data_ptr()), c_void_p(primals_684.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf801.data_ptr()), c_void_p(buf817.data_ptr()), c_void_p(buf834.data_ptr()), c_void_p(buf788.data_ptr()))
    del primals_320
    # Source Nodes: [bottleneck_output_102], Original ATen: [aten.convolution]
    buf789 = extern_kernels.convolution(buf788, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf789, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf790 = reinterpret_tensor(buf278, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf278  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_107(c_void_p(buf789.data_ptr()), c_void_p(primals_686.data_ptr()), c_void_p(primals_687.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(buf790.data_ptr()))
    del primals_323
    # Source Nodes: [new_features_102], Original ATen: [aten.convolution]
    buf791 = extern_kernels.convolution(buf790, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf791, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf792 = reinterpret_tensor(buf803, (4, 512, 7, 7), (40768, 1, 5824, 832), 0)  # alias
    buf808 = reinterpret_tensor(buf820, (4, 512, 7, 7), (42336, 1, 6048, 864), 0)  # alias
    buf825 = reinterpret_tensor(buf838, (4, 512, 7, 7), (43904, 1, 6272, 896), 0)  # alias
    buf857 = empty_strided((4, 928, 7, 7), (45472, 1, 6496, 928), device='cpu', dtype=torch.float32)
    buf843 = reinterpret_tensor(buf857, (4, 512, 7, 7), (45472, 1, 6496, 928), 0)  # alias
    buf793 = reinterpret_tensor(buf803, (4, 32, 7, 7), (40768, 1, 5824, 832), 512)  # alias
    buf809 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 512)  # alias
    buf826 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 512)  # alias
    buf844 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 512)  # alias
    buf794 = reinterpret_tensor(buf803, (4, 32, 7, 7), (40768, 1, 5824, 832), 544)  # alias
    buf810 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 544)  # alias
    buf827 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 544)  # alias
    buf845 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 544)  # alias
    buf795 = reinterpret_tensor(buf803, (4, 32, 7, 7), (40768, 1, 5824, 832), 576)  # alias
    buf811 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 576)  # alias
    buf828 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 576)  # alias
    buf846 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 576)  # alias
    buf796 = reinterpret_tensor(buf803, (4, 32, 7, 7), (40768, 1, 5824, 832), 608)  # alias
    buf812 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 608)  # alias
    buf829 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 608)  # alias
    buf847 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 608)  # alias
    buf797 = reinterpret_tensor(buf803, (4, 32, 7, 7), (40768, 1, 5824, 832), 640)  # alias
    buf813 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 640)  # alias
    buf830 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 640)  # alias
    buf848 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 640)  # alias
    buf802 = reinterpret_tensor(buf803, (4, 32, 7, 7), (40768, 1, 5824, 832), 800)  # alias
    buf818 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 800)  # alias
    buf835 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 800)  # alias
    buf853 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 800)  # alias
    buf804 = empty_strided((4, 832, 7, 7), (40768, 1, 5824, 832), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_108(c_void_p(buf697.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf791.data_ptr()), c_void_p(buf803.data_ptr()), c_void_p(primals_689.data_ptr()), c_void_p(primals_690.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf825.data_ptr()), c_void_p(buf843.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf809.data_ptr()), c_void_p(buf826.data_ptr()), c_void_p(buf844.data_ptr()), c_void_p(buf794.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf845.data_ptr()), c_void_p(buf795.data_ptr()), c_void_p(buf811.data_ptr()), c_void_p(buf828.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(buf796.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(buf829.data_ptr()), c_void_p(buf847.data_ptr()), c_void_p(buf797.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf830.data_ptr()), c_void_p(buf848.data_ptr()), c_void_p(buf802.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(buf835.data_ptr()), c_void_p(buf853.data_ptr()), c_void_p(buf804.data_ptr()))
    del primals_326
    # Source Nodes: [bottleneck_output_104], Original ATen: [aten.convolution]
    buf805 = extern_kernels.convolution(buf804, primals_327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf805, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf806 = reinterpret_tensor(buf273, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf273  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_109(c_void_p(buf805.data_ptr()), c_void_p(primals_692.data_ptr()), c_void_p(primals_693.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(buf806.data_ptr()))
    del primals_329
    # Source Nodes: [new_features_104], Original ATen: [aten.convolution]
    buf807 = extern_kernels.convolution(buf806, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf807, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf814 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 672)  # alias
    buf831 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 672)  # alias
    buf849 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 672)  # alias
    buf877 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf868 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 672)  # alias
    buf819 = reinterpret_tensor(buf820, (4, 32, 7, 7), (42336, 1, 6048, 864), 832)  # alias
    buf836 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 832)  # alias
    buf854 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 832)  # alias
    buf873 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 832)  # alias
    buf821 = empty_strided((4, 864, 7, 7), (42336, 1, 6048, 864), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_110(c_void_p(buf737.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf820.data_ptr()), c_void_p(primals_695.data_ptr()), c_void_p(primals_696.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf831.data_ptr()), c_void_p(buf849.data_ptr()), c_void_p(buf868.data_ptr()), c_void_p(buf819.data_ptr()), c_void_p(buf836.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(buf873.data_ptr()), c_void_p(buf821.data_ptr()))
    del primals_332
    # Source Nodes: [bottleneck_output_106], Original ATen: [aten.convolution]
    buf822 = extern_kernels.convolution(buf821, primals_333, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf822, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf823 = reinterpret_tensor(buf268, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf268  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_111(c_void_p(buf822.data_ptr()), c_void_p(primals_698.data_ptr()), c_void_p(primals_699.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(buf823.data_ptr()))
    del primals_335
    # Source Nodes: [new_features_106], Original ATen: [aten.convolution]
    buf824 = extern_kernels.convolution(buf823, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf824, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf832 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 704)  # alias
    buf850 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 704)  # alias
    buf869 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 704)  # alias
    buf898 = empty_strided((4, 992, 7, 7), (48608, 1, 6944, 992), device='cpu', dtype=torch.float32)
    buf889 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 704)  # alias
    buf837 = reinterpret_tensor(buf838, (4, 32, 7, 7), (43904, 1, 6272, 896), 864)  # alias
    buf855 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 864)  # alias
    buf874 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 864)  # alias
    buf894 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 864)  # alias
    buf839 = empty_strided((4, 896, 7, 7), (43904, 1, 6272, 896), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_112(c_void_p(buf749.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf838.data_ptr()), c_void_p(primals_701.data_ptr()), c_void_p(primals_702.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(buf832.data_ptr()), c_void_p(buf850.data_ptr()), c_void_p(buf869.data_ptr()), c_void_p(buf889.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(buf855.data_ptr()), c_void_p(buf874.data_ptr()), c_void_p(buf894.data_ptr()), c_void_p(buf839.data_ptr()))
    del primals_338
    # Source Nodes: [bottleneck_output_108], Original ATen: [aten.convolution]
    buf840 = extern_kernels.convolution(buf839, primals_339, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf840, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf841 = reinterpret_tensor(buf263, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf263  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_113(c_void_p(buf840.data_ptr()), c_void_p(primals_704.data_ptr()), c_void_p(primals_705.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(buf841.data_ptr()))
    del primals_341
    # Source Nodes: [new_features_108], Original ATen: [aten.convolution]
    buf842 = extern_kernels.convolution(buf841, buf55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf842, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf851 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 736)  # alias
    buf870 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 736)  # alias
    buf890 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 736)  # alias
    buf920 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    buf911 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 736)  # alias
    buf852 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 768)  # alias
    buf871 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 768)  # alias
    buf891 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 768)  # alias
    buf912 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 768)  # alias
    buf856 = reinterpret_tensor(buf857, (4, 32, 7, 7), (45472, 1, 6496, 928), 896)  # alias
    buf875 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 896)  # alias
    buf895 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 896)  # alias
    buf916 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 896)  # alias
    buf858 = empty_strided((4, 928, 7, 7), (45472, 1, 6496, 928), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_114(c_void_p(buf762.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(buf842.data_ptr()), c_void_p(buf857.data_ptr()), c_void_p(primals_707.data_ptr()), c_void_p(primals_708.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(buf851.data_ptr()), c_void_p(buf870.data_ptr()), c_void_p(buf890.data_ptr()), c_void_p(buf911.data_ptr()), c_void_p(buf852.data_ptr()), c_void_p(buf871.data_ptr()), c_void_p(buf891.data_ptr()), c_void_p(buf912.data_ptr()), c_void_p(buf856.data_ptr()), c_void_p(buf875.data_ptr()), c_void_p(buf895.data_ptr()), c_void_p(buf916.data_ptr()), c_void_p(buf858.data_ptr()))
    del buf762
    del buf776
    del buf842
    del primals_344
    # Source Nodes: [bottleneck_output_110], Original ATen: [aten.convolution]
    buf859 = extern_kernels.convolution(buf858, primals_345, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf859, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf860 = reinterpret_tensor(buf639, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf639  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_115(c_void_p(buf859.data_ptr()), c_void_p(primals_710.data_ptr()), c_void_p(primals_711.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(buf860.data_ptr()))
    del primals_347
    # Source Nodes: [new_features_110], Original ATen: [aten.convolution]
    buf861 = extern_kernels.convolution(buf860, buf56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf861, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf862 = reinterpret_tensor(buf877, (4, 512, 7, 7), (47040, 1, 6720, 960), 0)  # alias
    buf882 = reinterpret_tensor(buf898, (4, 512, 7, 7), (48608, 1, 6944, 992), 0)  # alias
    buf903 = reinterpret_tensor(buf920, (4, 512, 7, 7), (50176, 1, 7168, 1024), 0)  # alias
    buf863 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 512)  # alias
    buf883 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 512)  # alias
    buf904 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 512)  # alias
    buf864 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 544)  # alias
    buf884 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 544)  # alias
    buf905 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 544)  # alias
    buf865 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 576)  # alias
    buf885 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 576)  # alias
    buf906 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 576)  # alias
    buf866 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 608)  # alias
    buf886 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 608)  # alias
    buf907 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 608)  # alias
    buf867 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 640)  # alias
    buf887 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 640)  # alias
    buf908 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 640)  # alias
    buf872 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 800)  # alias
    buf892 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 800)  # alias
    buf913 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 800)  # alias
    buf876 = reinterpret_tensor(buf877, (4, 32, 7, 7), (47040, 1, 6720, 960), 928)  # alias
    buf896 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 928)  # alias
    buf917 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 928)  # alias
    buf878 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_116(c_void_p(buf697.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf791.data_ptr()), c_void_p(buf861.data_ptr()), c_void_p(buf877.data_ptr()), c_void_p(primals_713.data_ptr()), c_void_p(primals_714.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(buf862.data_ptr()), c_void_p(buf882.data_ptr()), c_void_p(buf903.data_ptr()), c_void_p(buf863.data_ptr()), c_void_p(buf883.data_ptr()), c_void_p(buf904.data_ptr()), c_void_p(buf864.data_ptr()), c_void_p(buf884.data_ptr()), c_void_p(buf905.data_ptr()), c_void_p(buf865.data_ptr()), c_void_p(buf885.data_ptr()), c_void_p(buf906.data_ptr()), c_void_p(buf866.data_ptr()), c_void_p(buf886.data_ptr()), c_void_p(buf907.data_ptr()), c_void_p(buf867.data_ptr()), c_void_p(buf887.data_ptr()), c_void_p(buf908.data_ptr()), c_void_p(buf872.data_ptr()), c_void_p(buf892.data_ptr()), c_void_p(buf913.data_ptr()), c_void_p(buf876.data_ptr()), c_void_p(buf896.data_ptr()), c_void_p(buf917.data_ptr()), c_void_p(buf878.data_ptr()))
    del buf701
    del buf706
    del buf711
    del buf716
    del buf726
    del buf791
    del buf861
    del primals_350
    # Source Nodes: [bottleneck_output_112], Original ATen: [aten.convolution]
    buf879 = extern_kernels.convolution(buf878, primals_351, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf879, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf880 = reinterpret_tensor(buf558, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf558  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_117(c_void_p(buf879.data_ptr()), c_void_p(primals_716.data_ptr()), c_void_p(primals_717.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(buf880.data_ptr()))
    del primals_353
    # Source Nodes: [new_features_112], Original ATen: [aten.convolution]
    buf881 = extern_kernels.convolution(buf880, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf881, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf888 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 672)  # alias
    buf909 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 672)  # alias
    buf893 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 832)  # alias
    buf914 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 832)  # alias
    buf897 = reinterpret_tensor(buf898, (4, 32, 7, 7), (48608, 1, 6944, 992), 960)  # alias
    buf918 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 960)  # alias
    buf899 = empty_strided((4, 992, 7, 7), (48608, 1, 6944, 992), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_118(c_void_p(buf737.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf881.data_ptr()), c_void_p(buf898.data_ptr()), c_void_p(primals_719.data_ptr()), c_void_p(primals_720.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(buf888.data_ptr()), c_void_p(buf909.data_ptr()), c_void_p(buf893.data_ptr()), c_void_p(buf914.data_ptr()), c_void_p(buf897.data_ptr()), c_void_p(buf918.data_ptr()), c_void_p(buf899.data_ptr()))
    del buf737
    del buf807
    del buf881
    del primals_356
    # Source Nodes: [bottleneck_output_114], Original ATen: [aten.convolution]
    buf900 = extern_kernels.convolution(buf899, primals_357, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf900, (4, 128, 7, 7), (6272, 1, 896, 128))
    buf901 = reinterpret_tensor(buf486, (4, 128, 7, 7), (6272, 1, 896, 128), 0); del buf486  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_119(c_void_p(buf900.data_ptr()), c_void_p(primals_722.data_ptr()), c_void_p(primals_723.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(buf901.data_ptr()))
    del primals_359
    # Source Nodes: [new_features_114], Original ATen: [aten.convolution]
    buf902 = extern_kernels.convolution(buf901, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf902, (4, 32, 7, 7), (1568, 1, 224, 32))
    buf910 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 704)  # alias
    buf915 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 864)  # alias
    buf919 = reinterpret_tensor(buf920, (4, 32, 7, 7), (50176, 1, 7168, 1024), 992)  # alias
    buf921 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    buf922 = empty_strided((4, 1024, 1, 1), (1024, 1, 4096, 4096), device='cpu', dtype=torch.float32)
    buf923 = reinterpret_tensor(buf922, (4, 1024), (1024, 1), 0); del buf922  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_relu_view_120(c_void_p(buf923.data_ptr()), c_void_p(buf749.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf902.data_ptr()), c_void_p(buf920.data_ptr()), c_void_p(primals_725.data_ptr()), c_void_p(primals_726.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(buf910.data_ptr()), c_void_p(buf915.data_ptr()), c_void_p(buf919.data_ptr()), c_void_p(buf921.data_ptr()))
    del buf749
    del buf824
    del buf902
    del primals_362
    buf924 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_364, buf923, reinterpret_tensor(primals_363, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf924)
    del primals_364
    buf925 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.bool)
    buf926 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_threshold_backward_121(c_void_p(buf921.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(buf925.data_ptr()), c_void_p(buf926.data_ptr()))
    del buf921
    del primals_368
    return (buf924, buf0, primals_2, primals_4, primals_6, primals_7, buf1, primals_10, primals_12, primals_13, buf2, primals_16, primals_18, primals_19, buf3, primals_22, primals_24, primals_25, buf4, primals_28, primals_30, primals_31, buf5, primals_34, primals_36, primals_37, buf6, primals_40, primals_42, primals_43, primals_45, primals_46, buf7, primals_49, primals_51, primals_52, buf8, primals_55, primals_57, primals_58, buf9, primals_61, primals_63, primals_64, buf10, primals_67, primals_69, primals_70, buf11, primals_73, primals_75, primals_76, buf12, primals_79, primals_81, primals_82, buf13, primals_85, primals_87, primals_88, buf14, primals_91, primals_93, primals_94, buf15, primals_97, primals_99, primals_100, buf16, primals_103, primals_105, primals_106, buf17, primals_109, primals_111, primals_112, buf18, primals_115, primals_117, primals_118, primals_120, primals_121, buf19, primals_124, primals_126, primals_127, buf20, primals_130, primals_132, primals_133, buf21, primals_136, primals_138, primals_139, buf22, primals_142, primals_144, primals_145, buf23, primals_148, primals_150, primals_151, buf24, primals_154, primals_156, primals_157, buf25, primals_160, primals_162, primals_163, buf26, primals_166, primals_168, primals_169, buf27, primals_172, primals_174, primals_175, buf28, primals_178, primals_180, primals_181, buf29, primals_184, primals_186, primals_187, buf30, primals_190, primals_192, primals_193, buf31, primals_196, primals_198, primals_199, buf32, primals_202, primals_204, primals_205, buf33, primals_208, primals_210, primals_211, buf34, primals_214, primals_216, primals_217, buf35, primals_220, primals_222, primals_223, buf36, primals_226, primals_228, primals_229, buf37, primals_232, primals_234, primals_235, buf38, primals_238, primals_240, primals_241, buf39, primals_244, primals_246, primals_247, buf40, primals_250, primals_252, primals_253, buf41, primals_256, primals_258, primals_259, buf42, primals_262, primals_264, primals_265, primals_267, primals_268, buf43, primals_271, primals_273, primals_274, buf44, primals_277, primals_279, primals_280, buf45, primals_283, primals_285, primals_286, buf46, primals_289, primals_291, primals_292, buf47, primals_295, primals_297, primals_298, buf48, primals_301, primals_303, primals_304, buf49, primals_307, primals_309, primals_310, buf50, primals_313, primals_315, primals_316, buf51, primals_319, primals_321, primals_322, buf52, primals_325, primals_327, primals_328, buf53, primals_331, primals_333, primals_334, buf54, primals_337, primals_339, primals_340, buf55, primals_343, primals_345, primals_346, buf56, primals_349, primals_351, primals_352, buf57, primals_355, primals_357, primals_358, buf58, primals_361, primals_365, primals_366, primals_369, primals_371, primals_372, primals_374, primals_375, primals_377, primals_378, primals_380, primals_381, primals_383, primals_384, primals_386, primals_387, primals_389, primals_390, primals_392, primals_393, primals_395, primals_396, primals_398, primals_399, primals_401, primals_402, primals_404, primals_405, primals_407, primals_408, primals_410, primals_411, primals_413, primals_414, primals_416, primals_417, primals_419, primals_420, primals_422, primals_423, primals_425, primals_426, primals_428, primals_429, primals_431, primals_432, primals_434, primals_435, primals_437, primals_438, primals_440, primals_441, primals_443, primals_444, primals_446, primals_447, primals_449, primals_450, primals_452, primals_453, primals_455, primals_456, primals_458, primals_459, primals_461, primals_462, primals_464, primals_465, primals_467, primals_468, primals_470, primals_471, primals_473, primals_474, primals_476, primals_477, primals_479, primals_480, primals_482, primals_483, primals_485, primals_486, primals_488, primals_489, primals_491, primals_492, primals_494, primals_495, primals_497, primals_498, primals_500, primals_501, primals_503, primals_504, primals_506, primals_507, primals_509, primals_510, primals_512, primals_513, primals_515, primals_516, primals_518, primals_519, primals_521, primals_522, primals_524, primals_525, primals_527, primals_528, primals_530, primals_531, primals_533, primals_534, primals_536, primals_537, primals_539, primals_540, primals_542, primals_543, primals_545, primals_546, primals_548, primals_549, primals_551, primals_552, primals_554, primals_555, primals_557, primals_558, primals_560, primals_561, primals_563, primals_564, primals_566, primals_567, primals_569, primals_570, primals_572, primals_573, primals_575, primals_576, primals_578, primals_579, primals_581, primals_582, primals_584, primals_585, primals_587, primals_588, primals_590, primals_591, primals_593, primals_594, primals_596, primals_597, primals_599, primals_600, primals_602, primals_603, primals_605, primals_606, primals_608, primals_609, primals_611, primals_612, primals_614, primals_615, primals_617, primals_618, primals_620, primals_621, primals_623, primals_624, primals_626, primals_627, primals_629, primals_630, primals_632, primals_633, primals_635, primals_636, primals_638, primals_639, primals_641, primals_642, primals_644, primals_645, primals_647, primals_648, primals_650, primals_651, primals_653, primals_654, primals_656, primals_657, primals_659, primals_660, primals_662, primals_663, primals_665, primals_666, primals_668, primals_669, primals_671, primals_672, primals_674, primals_675, primals_677, primals_678, primals_680, primals_681, primals_683, primals_684, primals_686, primals_687, primals_689, primals_690, primals_692, primals_693, primals_695, primals_696, primals_698, primals_699, primals_701, primals_702, primals_704, primals_705, primals_707, primals_708, primals_710, primals_711, primals_713, primals_714, primals_716, primals_717, primals_719, primals_720, primals_722, primals_723, primals_725, primals_726, buf59, buf60, buf61, buf63, buf64, buf65, buf66, buf69, buf70, buf71, buf72, buf74, buf75, buf76, buf77, buf79, buf80, buf81, buf82, buf89, buf90, buf91, buf92, buf100, buf101, buf102, buf103, buf112, buf113, buf114, buf115, buf116, buf117, buf118, buf120, buf121, buf122, buf123, buf125, buf126, buf127, buf128, buf130, buf131, buf132, buf133, buf140, buf141, buf142, buf143, buf151, buf152, buf153, buf154, buf163, buf164, buf165, buf166, buf176, buf177, buf178, buf179, buf190, buf191, buf192, buf193, buf205, buf206, buf207, buf208, buf221, buf222, buf223, buf224, buf238, buf239, buf240, buf241, buf256, buf257, buf258, buf259, buf260, buf261, buf262, buf264, buf265, buf266, buf267, buf269, buf270, buf271, buf272, buf274, buf275, buf276, buf277, buf284, buf285, buf286, buf287, buf295, buf296, buf297, buf298, buf307, buf308, buf309, buf310, buf320, buf321, buf322, buf323, buf334, buf335, buf336, buf337, buf349, buf350, buf351, buf352, buf365, buf366, buf367, buf368, buf382, buf383, buf384, buf385, buf400, buf401, buf402, buf403, buf419, buf420, buf421, buf422, buf439, buf440, buf441, buf442, buf460, buf461, buf462, buf463, buf482, buf483, buf484, buf485, buf505, buf506, buf507, buf508, buf529, buf530, buf531, buf532, buf554, buf555, buf556, buf557, buf580, buf581, buf582, buf583, buf607, buf608, buf609, buf610, buf635, buf636, buf637, buf638, buf664, buf665, buf666, buf667, buf694, buf695, buf696, buf697, buf698, buf699, buf700, buf702, buf703, buf704, buf705, buf707, buf708, buf709, buf710, buf712, buf713, buf714, buf715, buf722, buf723, buf724, buf725, buf733, buf734, buf735, buf736, buf745, buf746, buf747, buf748, buf758, buf759, buf760, buf761, buf772, buf773, buf774, buf775, buf787, buf788, buf789, buf790, buf803, buf804, buf805, buf806, buf820, buf821, buf822, buf823, buf838, buf839, buf840, buf841, buf857, buf858, buf859, buf860, buf877, buf878, buf879, buf880, buf898, buf899, buf900, buf901, buf920, buf923, reinterpret_tensor(primals_363, (1000, 1024), (1024, 1), 0), buf925, buf926, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((128, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((128, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((128, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((128, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((128, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((128, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((128, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((128, 352, 1, 1), (352, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((128, 416, 1, 1), (416, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((128, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((128, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((128, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((128, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((128, 352, 1, 1), (352, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((128, 416, 1, 1), (416, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((128, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((128, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((128, 544, 1, 1), (544, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((128, 608, 1, 1), (608, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((128, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((128, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((128, 704, 1, 1), (704, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((128, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((128, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((128, 832, 1, 1), (832, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((128, 864, 1, 1), (864, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((128, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((128, 928, 1, 1), (928, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((128, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((128, 992, 1, 1), (992, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((128, 544, 1, 1), (544, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((128, 608, 1, 1), (608, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((128, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((128, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((128, 704, 1, 1), (704, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((128, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((128, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((128, 832, 1, 1), (832, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((128, 864, 1, 1), (864, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((128, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((128, 928, 1, 1), (928, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_346 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_348 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_349 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_351 = rand_strided((128, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_353 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_354 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_355 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    primals_357 = rand_strided((128, 992, 1, 1), (992, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_358 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_359 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_360 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_361 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_362 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_363 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_364 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_365 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_366 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_367 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_368 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_369 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_370 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_371 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_372 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_373 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_374 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_375 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_376 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_377 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_378 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_379 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_380 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_381 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_382 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_383 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_384 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_385 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_386 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_387 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_388 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_389 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_390 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_391 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_392 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_393 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_394 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_395 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_396 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_397 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_398 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_399 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_400 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_401 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_402 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_403 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_404 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_405 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_406 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_407 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_408 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_409 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_410 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_411 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_412 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_413 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_414 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_415 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_416 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_417 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_418 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_419 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_420 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_421 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_422 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_423 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_424 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_425 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_426 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_427 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_428 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_429 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_430 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_431 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_432 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_433 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_434 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_435 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_436 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_437 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    primals_438 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    primals_439 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_440 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_441 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_442 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_443 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_444 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_445 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_446 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_447 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_448 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_449 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    primals_450 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    primals_451 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_452 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_453 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_454 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_455 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_456 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_457 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_458 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_459 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_460 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_461 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    primals_462 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    primals_463 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_464 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_465 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_466 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_467 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_468 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_469 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_470 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_471 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_472 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_473 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_474 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_475 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_476 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_477 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_478 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_479 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_480 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_481 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_482 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_483 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_484 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_485 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_486 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_487 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_488 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    primals_489 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    primals_490 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_491 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_492 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_493 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_494 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_495 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_496 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_497 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_498 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_499 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_500 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    primals_501 = rand_strided((352, ), (1, ), device='cpu', dtype=torch.float32)
    primals_502 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_503 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_504 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_505 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_506 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_507 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_508 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_509 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_510 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_511 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_512 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    primals_513 = rand_strided((416, ), (1, ), device='cpu', dtype=torch.float32)
    primals_514 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_515 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_516 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_517 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_518 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_519 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_520 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_521 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_522 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_523 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_524 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_525 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_526 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_527 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_528 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_529 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_530 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_531 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_532 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_533 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_534 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_535 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_536 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    primals_537 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    primals_538 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_539 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_540 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_541 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_542 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_543 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_544 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_545 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_546 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_547 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_548 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    primals_549 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    primals_550 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_551 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_552 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_553 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_554 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_555 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_556 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_557 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_558 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_559 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_560 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_561 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_562 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_563 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_564 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_565 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_566 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    primals_567 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    primals_568 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_569 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_570 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_571 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_572 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_573 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_574 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_575 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_576 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_577 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_578 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_579 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_580 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_581 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_582 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_583 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_584 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_585 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_586 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_587 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_588 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_589 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_590 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    primals_591 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    primals_592 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_593 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_594 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_595 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_596 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    primals_597 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    primals_598 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_599 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_600 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_601 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_602 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_603 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_604 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_605 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_606 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_607 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_608 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    primals_609 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    primals_610 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_611 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_612 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_613 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_614 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_615 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_616 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_617 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_618 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_619 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_620 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    primals_621 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    primals_622 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_623 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_624 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_625 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_626 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_627 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_628 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_629 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_630 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_631 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_632 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_633 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_634 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_635 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    primals_636 = rand_strided((544, ), (1, ), device='cpu', dtype=torch.float32)
    primals_637 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_638 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_639 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_640 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_641 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_642 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_643 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_644 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_645 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_646 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_647 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    primals_648 = rand_strided((608, ), (1, ), device='cpu', dtype=torch.float32)
    primals_649 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_650 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_651 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_652 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_653 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_654 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_655 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_656 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_657 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_658 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_659 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_660 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_661 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_662 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_663 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_664 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_665 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    primals_666 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    primals_667 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_668 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_669 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_670 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_671 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_672 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_673 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_674 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_675 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_676 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_677 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_678 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_679 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_680 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_681 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_682 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_683 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_684 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_685 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_686 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_687 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_688 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_689 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    primals_690 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    primals_691 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_692 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_693 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_694 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_695 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    primals_696 = rand_strided((864, ), (1, ), device='cpu', dtype=torch.float32)
    primals_697 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_698 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_699 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_700 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_701 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_702 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_703 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_704 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_705 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_706 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_707 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    primals_708 = rand_strided((928, ), (1, ), device='cpu', dtype=torch.float32)
    primals_709 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_710 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_711 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_712 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_713 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_714 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_715 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_716 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_717 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_718 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_719 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    primals_720 = rand_strided((992, ), (1, ), device='cpu', dtype=torch.float32)
    primals_721 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_722 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_723 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_724 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_725 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_726 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_727 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_728 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('densenet121', benchmark_compiled_module)
