
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
                       float* out_ptr11)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (512L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (32L*x2) + (512L*x0)), static_cast<long>(32L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (48L*x2) + (432L*x0)), static_cast<long>(48L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (48L*x2) + (432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (48L*x2) + (432L*x0)), static_cast<long>(48L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (48L*x2) + (432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (48L*x2) + (432L*x0)), static_cast<long>(48L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (48L*x2) + (432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (48L*x2) + (432L*x0)), static_cast<long>(48L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (48L*x2) + (432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(x1 + (48L*x2) + (432L*x0)), static_cast<long>(48L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr8[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (432L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (48L*x2) + (432L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr9[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (768L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (192L*x2) + (768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr10[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1536L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr10 + static_cast<long>(x1 + (384L*x2) + (1536L*x0)));
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
                        auto tmp0 = in_ptr11[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr11[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
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
                    auto tmp1 = static_cast<float>(100352.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
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


cpp_fused__native_batch_norm_legit_functional_add_2 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
                    auto tmp1 = static_cast<float>(6272.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (784L*x2)), static_cast<long>(784L), tmp16, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(6272.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr3 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr7 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_gelu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_5 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
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
                    auto tmp1 = static_cast<float>(6272.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp6 = static_cast<float>(6272.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp13 = tmp4 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_gelu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
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
                    auto tmp1 = static_cast<float>(6272.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(6272.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 + tmp12;
                    auto tmp14 = tmp13.rsqrt();
                    auto tmp15 = tmp6 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_gelu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = static_cast<float>(6272.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(6272.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 + tmp14;
                    auto tmp16 = tmp15.rsqrt();
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_gelu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_14 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_gelu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_17 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
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
                    auto tmp1 = static_cast<float>(6272.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp6 = static_cast<float>(6272.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp13 = tmp4 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_gelu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
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
                    auto tmp1 = static_cast<float>(6272.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(6272.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 + tmp12;
                    auto tmp14 = tmp13.rsqrt();
                    auto tmp15 = tmp6 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_gelu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_24 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2)), static_cast<long>(196L), tmp16, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1568.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x2)];
                        auto tmp3 = out_ptr1[static_cast<long>(x2)];
                        auto tmp10 = in_ptr1[static_cast<long>(x2)];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x1 + (196L*x2))];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp4 = static_cast<float>(1568.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12544L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (64L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (12544L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (12544L*(c10::div_floor_integer((x2 + x2_inner), 64L))) + (75264L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp6 = static_cast<float>(1568.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp13 = tmp4 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(1568.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 + tmp12;
                    auto tmp14 = tmp13.rsqrt();
                    auto tmp15 = tmp6 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12544L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (64L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (12544L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (12544L*(c10::div_floor_integer((x2 + x2_inner), 64L))) + (75264L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(1568.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 + tmp14;
                    auto tmp16 = tmp15.rsqrt();
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_36 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1)));
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
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12544L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (64L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (12544L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (12544L*(c10::div_floor_integer((x2 + x2_inner), 64L))) + (75264L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp6 = static_cast<float>(1568.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp13 = tmp4 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(1568.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 + tmp12;
                    auto tmp14 = tmp13.rsqrt();
                    auto tmp15 = tmp6 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12544L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9408L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (64L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (12544L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (12544L*(c10::div_floor_integer((x2 + x2_inner), 64L))) + (75264L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(1568.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 + tmp14;
                    auto tmp16 = tmp15.rsqrt();
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_49 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (49L*x2)), static_cast<long>(49L), tmp16, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(392.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr3 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x2)];
                        auto tmp3 = out_ptr1[static_cast<long>(x2)];
                        auto tmp10 = in_ptr1[static_cast<long>(x2)];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x1 + (49L*x2))];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp4 = static_cast<float>(392.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr3[static_cast<long>(x2 + (768L*x1) + (37632L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr7 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x1) + (2304L*x2) + (112896L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (128L*x2) + (6272L*x1) + (37632L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (112896L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (112896L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (49L*x0))];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1536L + x3 + (128L*x1) + (2304L*x2) + (112896L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (128L*x2) + (6272L*x1) + (37632L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 128L))) + (37632L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (37632L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp6 = static_cast<float>(392.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp13 = tmp4 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(392.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 + tmp12;
                    auto tmp14 = tmp13.rsqrt();
                    auto tmp15 = tmp6 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x1) + (2304L*x2) + (112896L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (128L*x2) + (6272L*x1) + (37632L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (112896L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (112896L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (49L*x0))];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1536L + x3 + (128L*x1) + (2304L*x2) + (112896L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (128L*x2) + (6272L*x1) + (37632L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 128L))) + (37632L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (37632L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(392.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 + tmp14;
                    auto tmp16 = tmp15.rsqrt();
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_61 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(301056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x1) + (2304L*x2) + (112896L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (128L*x2) + (6272L*x1) + (37632L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (112896L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (112896L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (49L*x0))];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1536L + x3 + (128L*x1) + (2304L*x2) + (112896L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (128L*x2) + (6272L*x1) + (37632L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 128L))) + (37632L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (37632L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp6 = static_cast<float>(392.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp13 = tmp4 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(392.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 + tmp12;
                    auto tmp14 = tmp13.rsqrt();
                    auto tmp15 = tmp6 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x1) + (2304L*x2) + (112896L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (128L*x2) + (6272L*x1) + (37632L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (112896L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (112896L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (49L*x0))];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.08838834764831845);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1536L + x3 + (128L*x1) + (2304L*x2) + (112896L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (128L*x2) + (6272L*x1) + (37632L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 128L))) + (37632L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (37632L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(392.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 + tmp14;
                    auto tmp16 = tmp15.rsqrt();
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_mean_view_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(301056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(392.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional__softmax_add_detach_74 = async_compile.cpp('''
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
                       const long* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const long* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const long* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const float* in_ptr30,
                       const long* in_ptr31,
                       const float* in_ptr32,
                       const float* in_ptr33,
                       const float* in_ptr34,
                       const long* in_ptr35,
                       const float* in_ptr36,
                       const float* in_ptr37,
                       const float* in_ptr38,
                       const long* in_ptr39,
                       const float* in_ptr40,
                       const float* in_ptr41,
                       const float* in_ptr42,
                       const long* in_ptr43,
                       const float* in_ptr44,
                       const float* in_ptr45,
                       const float* in_ptr46,
                       const long* in_ptr47,
                       const float* in_ptr48,
                       const float* in_ptr49,
                       const float* in_ptr50,
                       const long* in_ptr51,
                       const float* in_ptr52,
                       const float* in_ptr53,
                       const float* in_ptr54,
                       const long* in_ptr55,
                       const float* in_ptr56,
                       const float* in_ptr57,
                       const float* in_ptr58,
                       const long* in_ptr59,
                       const float* in_ptr60,
                       const float* in_ptr61,
                       const float* in_ptr62,
                       const long* in_ptr63,
                       const float* in_ptr64,
                       const float* in_ptr65,
                       const float* in_ptr66,
                       const long* in_ptr67,
                       const float* in_ptr68,
                       const float* in_ptr69,
                       const float* in_ptr70,
                       const long* in_ptr71,
                       const float* in_ptr72,
                       const float* in_ptr73,
                       const float* in_ptr74,
                       const long* in_ptr75,
                       const float* in_ptr76,
                       const float* in_ptr77,
                       const float* in_ptr78,
                       const long* in_ptr79,
                       const float* in_ptr80,
                       const float* in_ptr81,
                       const float* in_ptr82,
                       const long* in_ptr83,
                       const float* in_ptr84,
                       const float* in_ptr85,
                       const float* in_ptr86,
                       const long* in_ptr87,
                       const float* in_ptr88,
                       const float* in_ptr89,
                       const float* in_ptr90,
                       const long* in_ptr91,
                       const float* in_ptr92,
                       const float* in_ptr93,
                       const float* in_ptr94,
                       const long* in_ptr95,
                       const float* in_ptr96,
                       const float* in_ptr97,
                       const float* in_ptr98,
                       const long* in_ptr99,
                       const float* in_ptr100,
                       const float* in_ptr101,
                       const float* in_ptr102,
                       const long* in_ptr103,
                       const float* in_ptr104,
                       const float* in_ptr105,
                       const float* in_ptr106,
                       const long* in_ptr107,
                       const float* in_ptr108,
                       const float* in_ptr109,
                       const float* in_ptr110,
                       const long* in_ptr111,
                       const float* in_ptr112,
                       const float* in_ptr113,
                       const float* in_ptr114,
                       const long* in_ptr115,
                       const float* in_ptr116,
                       const float* in_ptr117,
                       const float* in_ptr118,
                       const long* in_ptr119,
                       const float* in_ptr120,
                       const float* in_ptr121,
                       const float* in_ptr122,
                       const long* in_ptr123,
                       const float* in_ptr124,
                       const float* in_ptr125,
                       const float* in_ptr126,
                       const long* in_ptr127,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
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
                       long* out_ptr147)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                            auto tmp1 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr0[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                            auto tmp1 = in_ptr3[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr1[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                            auto tmp1 = in_ptr5[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                            auto tmp1 = in_ptr5[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr2[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                            auto tmp1 = in_ptr7[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr3[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr6[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                            auto tmp1 = in_ptr7[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr3[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                            auto tmp1 = in_ptr9[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr4[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr8[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                            auto tmp1 = in_ptr9[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr4[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                            auto tmp1 = in_ptr11[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr5[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr10[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                            auto tmp1 = in_ptr11[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr5[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                            auto tmp1 = in_ptr13[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr6[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr12[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                            auto tmp1 = in_ptr13[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr6[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                            auto tmp1 = in_ptr15[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr7[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr14[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                            auto tmp1 = in_ptr15[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr7[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
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
                auto tmp0 = in_ptr19[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr12[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                auto tmp0 = in_ptr23[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr17[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                auto tmp0 = in_ptr27[static_cast<long>(0L)];
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr28 + static_cast<long>(x0));
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                auto tmp0 = in_ptr31[static_cast<long>(0L)];
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr32 + static_cast<long>(x0));
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                auto tmp0 = in_ptr35[static_cast<long>(0L)];
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr36 + static_cast<long>(x0));
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr35 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                auto tmp0 = in_ptr39[static_cast<long>(0L)];
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr40 + static_cast<long>(x0));
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr40 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                auto tmp0 = in_ptr43[static_cast<long>(0L)];
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr44 + static_cast<long>(x0));
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr45 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                auto tmp0 = in_ptr47[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr47[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr48 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr50 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                auto tmp0 = in_ptr51[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr52[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr52 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr55 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                auto tmp0 = in_ptr55[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr57[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr56 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr60 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                auto tmp0 = in_ptr59[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr62[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr60 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr65 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                auto tmp0 = in_ptr63[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr67[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr64 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr70 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                auto tmp0 = in_ptr67[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr72[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr68 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr75 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                auto tmp0 = in_ptr71[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr77[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr72 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr80 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                auto tmp0 = in_ptr75[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr82[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr76 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr85 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                auto tmp0 = in_ptr79[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr87[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr80 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr90 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                auto tmp0 = in_ptr83[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr92[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr84 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr95 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                auto tmp0 = in_ptr87[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr97[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr88 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr100 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                auto tmp0 = in_ptr91[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr102[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr92 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr105 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                auto tmp0 = in_ptr95[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr107[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr96 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr110 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                auto tmp0 = in_ptr99[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr112[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr100 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr115 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                auto tmp0 = in_ptr103[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr117[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr104 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr120 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                auto tmp0 = in_ptr107[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr122[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr108 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr125 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                auto tmp0 = in_ptr111[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr127[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr112 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr130 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                auto tmp0 = in_ptr115[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr132[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr116 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr135 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                auto tmp0 = in_ptr119[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr137[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr120 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr140 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                auto tmp0 = in_ptr123[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr142[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr124 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr145 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                auto tmp0 = in_ptr127[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr147[static_cast<long>(0L)] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206 = args
    args.clear()
    assert_size_stride(primals_1, (1, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(primals_2, (1, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(primals_3, (1, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(primals_4, (32, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (192, 32, 4, 4), (512, 16, 4, 1))
    assert_size_stride(primals_8, (192, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_10, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_12, (192, ), (1, ))
    assert_size_stride(primals_13, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_14, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_15, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_16, (192, ), (1, ))
    assert_size_stride(primals_17, (192, ), (1, ))
    assert_size_stride(primals_18, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_19, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_20, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_22, (192, ), (1, ))
    assert_size_stride(primals_23, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_24, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_25, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_28, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_29, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_30, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_34, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_35, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_36, (192, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_38, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_39, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_40, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_42, (192, ), (1, ))
    assert_size_stride(primals_43, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_44, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_45, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_46, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(primals_47, (384, ), (1, ))
    assert_size_stride(primals_48, (384, ), (1, ))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_50, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_52, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_53, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_56, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_57, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_59, (384, ), (1, ))
    assert_size_stride(primals_60, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_61, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_65, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_69, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_71, (384, ), (1, ))
    assert_size_stride(primals_72, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_73, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_77, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_80, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_81, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_82, (768, 384, 2, 2), (1536, 4, 2, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_89, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_93, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_97, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_101, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_105, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_109, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_113, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_117, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (1000, 768), (768, 1))
    assert_size_stride(primals_121, (1000, ), (1, ))
    assert_size_stride(primals_122, (32, ), (1, ))
    assert_size_stride(primals_123, (32, ), (1, ))
    assert_size_stride(primals_124, (), ())
    assert_size_stride(primals_125, (192, ), (1, ))
    assert_size_stride(primals_126, (192, ), (1, ))
    assert_size_stride(primals_127, (), ())
    assert_size_stride(primals_128, (192, ), (1, ))
    assert_size_stride(primals_129, (192, ), (1, ))
    assert_size_stride(primals_130, (), ())
    assert_size_stride(primals_131, (192, ), (1, ))
    assert_size_stride(primals_132, (192, ), (1, ))
    assert_size_stride(primals_133, (), ())
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_135, (192, ), (1, ))
    assert_size_stride(primals_136, (), ())
    assert_size_stride(primals_137, (192, ), (1, ))
    assert_size_stride(primals_138, (192, ), (1, ))
    assert_size_stride(primals_139, (), ())
    assert_size_stride(primals_140, (192, ), (1, ))
    assert_size_stride(primals_141, (192, ), (1, ))
    assert_size_stride(primals_142, (), ())
    assert_size_stride(primals_143, (192, ), (1, ))
    assert_size_stride(primals_144, (192, ), (1, ))
    assert_size_stride(primals_145, (), ())
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_147, (192, ), (1, ))
    assert_size_stride(primals_148, (), ())
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (), ())
    assert_size_stride(primals_152, (384, ), (1, ))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (), ())
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (384, ), (1, ))
    assert_size_stride(primals_157, (), ())
    assert_size_stride(primals_158, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (), ())
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (), ())
    assert_size_stride(primals_164, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (), ())
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (), ())
    assert_size_stride(primals_170, (384, ), (1, ))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_172, (), ())
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (384, ), (1, ))
    assert_size_stride(primals_175, (), ())
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (768, ), (1, ))
    assert_size_stride(primals_178, (), ())
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_181, (), ())
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_184, (), ())
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (), ())
    assert_size_stride(primals_188, (768, ), (1, ))
    assert_size_stride(primals_189, (768, ), (1, ))
    assert_size_stride(primals_190, (), ())
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (), ())
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_196, (), ())
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (), ())
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_202, (), ())
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (), ())
    assert_size_stride(primals_206, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((32, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((192, 32, 4, 4), (512, 1, 128, 32), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((384, 192, 2, 2), (768, 1, 384, 192), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((768, 384, 2, 2), (1536, 1, 768, 384), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_4.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    del primals_14
    del primals_19
    del primals_206
    del primals_24
    del primals_29
    del primals_34
    del primals_39
    del primals_4
    del primals_44
    del primals_46
    del primals_7
    del primals_82
    # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
    buf12 = extern_kernels.convolution(buf11, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf12, (8, 32, 112, 112), (401408, 1, 3584, 32))
    buf13 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf16 = empty((32, ), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_1(c_void_p(buf12.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    del primals_6
    # Source Nodes: [x_1], Original ATen: [aten.convolution]
    buf18 = extern_kernels.convolution(buf17, buf1, primals_8, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf18, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del primals_8
    buf19 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf22 = empty((192, ), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf27 = empty((192, ), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_2(c_void_p(buf18.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    del primals_1
    del primals_10
    del primals_12
    # Source Nodes: [x_5], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(buf28, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf29, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf30 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_3(c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    # Source Nodes: [x_8], Original ATen: [aten.convolution]
    buf31 = extern_kernels.convolution(buf30, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf31, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf32 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_4(c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    # Source Nodes: [x_10], Original ATen: [aten.convolution]
    buf33 = extern_kernels.convolution(buf32, primals_15, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf33, (8, 192, 28, 28), (150528, 1, 5376, 192))
    buf34 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf35 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf37 = empty((192, ), device='cpu', dtype=torch.float32)
    buf38 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_5(c_void_p(buf23.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    del primals_17
    # Source Nodes: [x_13], Original ATen: [aten.convolution]
    buf39 = extern_kernels.convolution(buf38, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf39, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf40 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_6(c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    # Source Nodes: [x_16], Original ATen: [aten.convolution]
    buf41 = extern_kernels.convolution(buf40, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf41, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf42 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_7(c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()))
    # Source Nodes: [x_18], Original ATen: [aten.convolution]
    buf43 = extern_kernels.convolution(buf42, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf43, (8, 192, 28, 28), (150528, 1, 5376, 192))
    buf44 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf45 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf47 = empty((192, ), device='cpu', dtype=torch.float32)
    buf48 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_8(c_void_p(buf23.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del primals_22
    # Source Nodes: [x_21], Original ATen: [aten.convolution]
    buf49 = extern_kernels.convolution(buf48, primals_23, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf50 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_9(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    # Source Nodes: [x_24], Original ATen: [aten.convolution]
    buf51 = extern_kernels.convolution(buf50, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf51, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf52 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_10(c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    # Source Nodes: [x_26], Original ATen: [aten.convolution]
    buf53 = extern_kernels.convolution(buf52, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf53, (8, 192, 28, 28), (150528, 1, 5376, 192))
    buf54 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf55 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf57 = empty((192, ), device='cpu', dtype=torch.float32)
    buf58 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_11(c_void_p(buf23.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    del primals_27
    # Source Nodes: [x_29], Original ATen: [aten.convolution]
    buf59 = extern_kernels.convolution(buf58, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf59, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf60 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_12(c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    # Source Nodes: [x_32], Original ATen: [aten.convolution]
    buf61 = extern_kernels.convolution(buf60, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf61, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf62 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_13(c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    # Source Nodes: [x_34], Original ATen: [aten.convolution]
    buf63 = extern_kernels.convolution(buf62, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf63, (8, 192, 28, 28), (150528, 1, 5376, 192))
    buf64 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    buf65 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf66 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf68 = empty((192, ), device='cpu', dtype=torch.float32)
    buf69 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_14(c_void_p(buf23.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()))
    del primals_32
    # Source Nodes: [x_37], Original ATen: [aten.convolution]
    buf70 = extern_kernels.convolution(buf69, primals_33, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf70, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf71 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_15(c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()))
    # Source Nodes: [x_40], Original ATen: [aten.convolution]
    buf72 = extern_kernels.convolution(buf71, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf72, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf73 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_16(c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    # Source Nodes: [x_42], Original ATen: [aten.convolution]
    buf74 = extern_kernels.convolution(buf73, primals_35, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf74, (8, 192, 28, 28), (150528, 1, 5376, 192))
    buf75 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf76 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf78 = empty((192, ), device='cpu', dtype=torch.float32)
    buf79 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_17(c_void_p(buf64.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del primals_37
    # Source Nodes: [x_45], Original ATen: [aten.convolution]
    buf80 = extern_kernels.convolution(buf79, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf80, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf81 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_18(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    # Source Nodes: [x_48], Original ATen: [aten.convolution]
    buf82 = extern_kernels.convolution(buf81, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf82, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf83 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_19(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    # Source Nodes: [x_50], Original ATen: [aten.convolution]
    buf84 = extern_kernels.convolution(buf83, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (8, 192, 28, 28), (150528, 1, 5376, 192))
    buf85 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf86 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf88 = empty((192, ), device='cpu', dtype=torch.float32)
    buf89 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_20(c_void_p(buf64.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    del primals_42
    # Source Nodes: [x_53], Original ATen: [aten.convolution]
    buf90 = extern_kernels.convolution(buf89, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf90, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf91 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_21(c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    # Source Nodes: [x_56], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf92, (8, 384, 28, 28), (301056, 1, 10752, 384))
    buf93 = empty_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_22(c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    # Source Nodes: [x_58], Original ATen: [aten.convolution]
    buf94 = extern_kernels.convolution(buf93, primals_45, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf94, (8, 192, 28, 28), (150528, 1, 5376, 192))
    buf95 = buf94; del buf94  # reuse
    cpp_fused_add_23(c_void_p(buf95.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf84.data_ptr()))
    # Source Nodes: [x_62], Original ATen: [aten.convolution]
    buf96 = extern_kernels.convolution(buf95, buf9, primals_47, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf96, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_47
    buf97 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf98 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf100 = empty((384, ), device='cpu', dtype=torch.float32)
    buf101 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf102 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf103 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf105 = empty((384, ), device='cpu', dtype=torch.float32)
    buf106 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_24(c_void_p(buf96.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del primals_2
    del primals_49
    del primals_51
    # Source Nodes: [getattr_l__mod___stage2___0___attn_qkv], Original ATen: [aten.convolution]
    buf107 = extern_kernels.convolution(buf106, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf107, (8, 1152, 14, 14), (225792, 1, 16128, 1152))
    buf108 = empty((8, 6, 196, 64), device='cpu', dtype=torch.float32)
    buf109 = empty((8, 6, 64, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_25(c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    buf110 = empty((48, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf108, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf109, (48, 64, 196), (12544, 196, 1), 0), out=buf110)
    buf111 = empty_strided((8, 6, 196, 1), (1176, 196, 1, 9408), device='cpu', dtype=torch.float32)
    buf112 = reinterpret_tensor(buf110, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf110  # reuse
    buf113 = empty_strided((8, 6, 196, 1), (1176, 196, 1, 9408), device='cpu', dtype=torch.float32)
    buf114 = empty((8, 6, 196, 196), device='cpu', dtype=torch.float32)
    buf115 = empty((8, 6, 196, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_26(c_void_p(buf112.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    del buf107
    buf116 = empty((48, 196, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_67], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf114, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf115, (48, 196, 64), (12544, 64, 1), 0), out=buf116)
    buf117 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_27(c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    # Source Nodes: [x_69], Original ATen: [aten.convolution]
    buf118 = extern_kernels.convolution(buf117, primals_53, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf118, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf119 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf120 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf122 = empty((384, ), device='cpu', dtype=torch.float32)
    buf123 = reinterpret_tensor(buf116, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf116  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_28(c_void_p(buf101.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    del primals_55
    # Source Nodes: [x_72], Original ATen: [aten.convolution]
    buf124 = extern_kernels.convolution(buf123, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    buf125 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_29(c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    # Source Nodes: [x_75], Original ATen: [aten.convolution]
    buf126 = extern_kernels.convolution(buf125, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf126, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf127 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf128 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf130 = empty((384, ), device='cpu', dtype=torch.float32)
    buf131 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_30(c_void_p(buf101.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del primals_59
    # Source Nodes: [getattr_l__mod___stage2___1___attn_qkv], Original ATen: [aten.convolution]
    buf132 = extern_kernels.convolution(buf131, primals_60, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf132, (8, 1152, 14, 14), (225792, 1, 16128, 1152))
    buf133 = empty((8, 6, 196, 64), device='cpu', dtype=torch.float32)
    buf134 = empty((8, 6, 64, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_31(c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()))
    buf135 = empty((48, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf133, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf134, (48, 64, 196), (12544, 196, 1), 0), out=buf135)
    buf136 = buf111; del buf111  # reuse
    buf137 = reinterpret_tensor(buf135, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf135  # reuse
    buf138 = empty_strided((8, 6, 196, 1), (1176, 196, 1, 9408), device='cpu', dtype=torch.float32)
    buf139 = empty((8, 6, 196, 196), device='cpu', dtype=torch.float32)
    buf140 = empty((8, 6, 196, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_32(c_void_p(buf137.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()))
    del buf132
    buf141 = empty((48, 196, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_79], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf139, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf140, (48, 196, 64), (12544, 64, 1), 0), out=buf141)
    buf142 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_33(c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    # Source Nodes: [x_81], Original ATen: [aten.convolution]
    buf143 = extern_kernels.convolution(buf142, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf143, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf144 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf145 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf147 = empty((384, ), device='cpu', dtype=torch.float32)
    buf148 = reinterpret_tensor(buf141, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf141  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_34(c_void_p(buf101.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    del primals_63
    # Source Nodes: [x_84], Original ATen: [aten.convolution]
    buf149 = extern_kernels.convolution(buf148, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf149, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    buf150 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_35(c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    # Source Nodes: [x_87], Original ATen: [aten.convolution]
    buf151 = extern_kernels.convolution(buf150, primals_65, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf151, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf152 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf153 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf154 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf156 = empty((384, ), device='cpu', dtype=torch.float32)
    buf157 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_36(c_void_p(buf101.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()))
    del primals_67
    # Source Nodes: [getattr_l__mod___stage2___2___attn_qkv], Original ATen: [aten.convolution]
    buf158 = extern_kernels.convolution(buf157, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf158, (8, 1152, 14, 14), (225792, 1, 16128, 1152))
    buf159 = empty((8, 6, 196, 64), device='cpu', dtype=torch.float32)
    buf160 = empty((8, 6, 64, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_37(c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    buf161 = empty((48, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf159, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf160, (48, 64, 196), (12544, 196, 1), 0), out=buf161)
    buf162 = buf136; del buf136  # reuse
    buf163 = reinterpret_tensor(buf161, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf161  # reuse
    buf164 = empty_strided((8, 6, 196, 1), (1176, 196, 1, 9408), device='cpu', dtype=torch.float32)
    buf165 = empty((8, 6, 196, 196), device='cpu', dtype=torch.float32)
    buf166 = empty((8, 6, 196, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_38(c_void_p(buf163.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    del buf158
    buf167 = empty((48, 196, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_91], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf165, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf166, (48, 196, 64), (12544, 64, 1), 0), out=buf167)
    buf168 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_39(c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    # Source Nodes: [x_93], Original ATen: [aten.convolution]
    buf169 = extern_kernels.convolution(buf168, primals_69, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf169, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf170 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf171 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf173 = empty((384, ), device='cpu', dtype=torch.float32)
    buf174 = reinterpret_tensor(buf167, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf167  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_40(c_void_p(buf152.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del primals_71
    # Source Nodes: [x_96], Original ATen: [aten.convolution]
    buf175 = extern_kernels.convolution(buf174, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf175, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    buf176 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_41(c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    # Source Nodes: [x_99], Original ATen: [aten.convolution]
    buf177 = extern_kernels.convolution(buf176, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf177, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf178 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf179 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf181 = empty((384, ), device='cpu', dtype=torch.float32)
    buf182 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_42(c_void_p(buf152.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()))
    del primals_75
    # Source Nodes: [getattr_l__mod___stage2___3___attn_qkv], Original ATen: [aten.convolution]
    buf183 = extern_kernels.convolution(buf182, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf183, (8, 1152, 14, 14), (225792, 1, 16128, 1152))
    buf184 = empty((8, 6, 196, 64), device='cpu', dtype=torch.float32)
    buf185 = empty((8, 6, 64, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_43(c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    buf186 = empty((48, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf184, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf185, (48, 64, 196), (12544, 196, 1), 0), out=buf186)
    buf187 = buf162; del buf162  # reuse
    buf188 = reinterpret_tensor(buf186, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf186  # reuse
    buf189 = empty_strided((8, 6, 196, 1), (1176, 196, 1, 9408), device='cpu', dtype=torch.float32)
    buf190 = empty((8, 6, 196, 196), device='cpu', dtype=torch.float32)
    buf191 = empty((8, 6, 196, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_44(c_void_p(buf188.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    del buf183
    del buf187
    buf192 = empty((48, 196, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_103], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf190, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf191, (48, 196, 64), (12544, 64, 1), 0), out=buf192)
    buf193 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_45(c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    # Source Nodes: [x_105], Original ATen: [aten.convolution]
    buf194 = extern_kernels.convolution(buf193, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf194, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf195 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf196 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf198 = empty((384, ), device='cpu', dtype=torch.float32)
    buf199 = reinterpret_tensor(buf192, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf192  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_46(c_void_p(buf152.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    del primals_79
    # Source Nodes: [x_108], Original ATen: [aten.convolution]
    buf200 = extern_kernels.convolution(buf199, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf200, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    buf201 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_47(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    # Source Nodes: [x_111], Original ATen: [aten.convolution]
    buf202 = extern_kernels.convolution(buf201, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf202, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf203 = buf202; del buf202  # reuse
    cpp_fused_add_48(c_void_p(buf203.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf194.data_ptr()))
    del buf152
    # Source Nodes: [x_115], Original ATen: [aten.convolution]
    buf204 = extern_kernels.convolution(buf203, buf10, primals_83, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf204, (8, 768, 7, 7), (37632, 1, 5376, 768))
    del primals_83
    buf205 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf206 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf208 = empty((768, ), device='cpu', dtype=torch.float32)
    buf209 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    buf210 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf213 = empty((768, ), device='cpu', dtype=torch.float32)
    buf214 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_49(c_void_p(buf204.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()))
    del primals_3
    del primals_85
    del primals_87
    # Source Nodes: [getattr_l__mod___stage3___0___attn_qkv], Original ATen: [aten.convolution]
    buf215 = extern_kernels.convolution(buf214, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf215, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
    buf216 = empty((8, 6, 49, 128), device='cpu', dtype=torch.float32)
    buf217 = empty((8, 6, 128, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_50(c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    buf218 = empty((48, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf216, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf217, (48, 128, 49), (6272, 49, 1), 0), out=buf218)
    buf219 = empty_strided((8, 6, 49, 1), (294, 49, 1, 2352), device='cpu', dtype=torch.float32)
    buf220 = reinterpret_tensor(buf218, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf218  # reuse
    buf221 = empty_strided((8, 6, 49, 1), (294, 49, 1, 2352), device='cpu', dtype=torch.float32)
    buf222 = empty((8, 6, 49, 49), device='cpu', dtype=torch.float32)
    buf223 = empty((8, 6, 49, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_51(c_void_p(buf220.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    del buf215
    buf224 = empty((48, 49, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_120], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf222, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf223, (48, 49, 128), (6272, 128, 1), 0), out=buf224)
    buf225 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_52(c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()))
    # Source Nodes: [x_122], Original ATen: [aten.convolution]
    buf226 = extern_kernels.convolution(buf225, primals_89, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf226, (8, 768, 7, 7), (37632, 1, 5376, 768))
    buf227 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf228 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf230 = empty((768, ), device='cpu', dtype=torch.float32)
    buf231 = reinterpret_tensor(buf224, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf224  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_53(c_void_p(buf209.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del primals_91
    # Source Nodes: [x_125], Original ATen: [aten.convolution]
    buf232 = extern_kernels.convolution(buf231, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf232, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    buf233 = reinterpret_tensor(buf64, (8, 3072, 7, 7), (150528, 1, 21504, 3072), 0); del buf64  # reuse
    cpp_fused_gelu_54(c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    # Source Nodes: [x_128], Original ATen: [aten.convolution]
    buf234 = extern_kernels.convolution(buf233, primals_93, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf234, (8, 768, 7, 7), (37632, 1, 5376, 768))
    buf235 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf236 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf238 = empty((768, ), device='cpu', dtype=torch.float32)
    buf239 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_55(c_void_p(buf209.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    del primals_95
    # Source Nodes: [getattr_l__mod___stage3___1___attn_qkv], Original ATen: [aten.convolution]
    buf240 = extern_kernels.convolution(buf239, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf240, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
    buf241 = empty((8, 6, 49, 128), device='cpu', dtype=torch.float32)
    buf242 = empty((8, 6, 128, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_56(c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    buf243 = empty((48, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf241, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf242, (48, 128, 49), (6272, 49, 1), 0), out=buf243)
    buf244 = buf219; del buf219  # reuse
    buf245 = reinterpret_tensor(buf243, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf243  # reuse
    buf246 = empty_strided((8, 6, 49, 1), (294, 49, 1, 2352), device='cpu', dtype=torch.float32)
    buf247 = empty((8, 6, 49, 49), device='cpu', dtype=torch.float32)
    buf248 = empty((8, 6, 49, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_57(c_void_p(buf245.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    del buf240
    buf249 = empty((48, 49, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_132], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf247, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf248, (48, 49, 128), (6272, 128, 1), 0), out=buf249)
    buf250 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_58(c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()))
    # Source Nodes: [x_134], Original ATen: [aten.convolution]
    buf251 = extern_kernels.convolution(buf250, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf251, (8, 768, 7, 7), (37632, 1, 5376, 768))
    buf252 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf253 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf255 = empty((768, ), device='cpu', dtype=torch.float32)
    buf256 = reinterpret_tensor(buf249, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf249  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_59(c_void_p(buf209.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    del primals_99
    # Source Nodes: [x_137], Original ATen: [aten.convolution]
    buf257 = extern_kernels.convolution(buf256, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf257, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    buf258 = empty_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_60(c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    # Source Nodes: [x_140], Original ATen: [aten.convolution]
    buf259 = extern_kernels.convolution(buf258, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf259, (8, 768, 7, 7), (37632, 1, 5376, 768))
    buf260 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    buf261 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf262 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf264 = empty((768, ), device='cpu', dtype=torch.float32)
    buf265 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_61(c_void_p(buf209.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del primals_103
    # Source Nodes: [getattr_l__mod___stage3___2___attn_qkv], Original ATen: [aten.convolution]
    buf266 = extern_kernels.convolution(buf265, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf266, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
    buf267 = empty((8, 6, 49, 128), device='cpu', dtype=torch.float32)
    buf268 = empty((8, 6, 128, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_62(c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = empty((48, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf267, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf268, (48, 128, 49), (6272, 49, 1), 0), out=buf269)
    buf270 = buf244; del buf244  # reuse
    buf271 = reinterpret_tensor(buf269, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf269  # reuse
    buf272 = empty_strided((8, 6, 49, 1), (294, 49, 1, 2352), device='cpu', dtype=torch.float32)
    buf273 = empty((8, 6, 49, 49), device='cpu', dtype=torch.float32)
    buf274 = empty((8, 6, 49, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_63(c_void_p(buf271.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del buf266
    buf275 = empty((48, 49, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_144], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf273, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf274, (48, 49, 128), (6272, 128, 1), 0), out=buf275)
    buf276 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_64(c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    # Source Nodes: [x_146], Original ATen: [aten.convolution]
    buf277 = extern_kernels.convolution(buf276, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf277, (8, 768, 7, 7), (37632, 1, 5376, 768))
    buf278 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf279 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf281 = empty((768, ), device='cpu', dtype=torch.float32)
    buf282 = reinterpret_tensor(buf275, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf275  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_65(c_void_p(buf260.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()))
    del primals_107
    # Source Nodes: [x_149], Original ATen: [aten.convolution]
    buf283 = extern_kernels.convolution(buf282, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf283, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    buf284 = empty_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_66(c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()))
    # Source Nodes: [x_152], Original ATen: [aten.convolution]
    buf285 = extern_kernels.convolution(buf284, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf285, (8, 768, 7, 7), (37632, 1, 5376, 768))
    buf286 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf287 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf289 = empty((768, ), device='cpu', dtype=torch.float32)
    buf290 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_67(c_void_p(buf260.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()))
    del primals_111
    # Source Nodes: [getattr_l__mod___stage3___3___attn_qkv], Original ATen: [aten.convolution]
    buf291 = extern_kernels.convolution(buf290, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf291, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
    buf292 = empty((8, 6, 49, 128), device='cpu', dtype=torch.float32)
    buf293 = empty((8, 6, 128, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_68(c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()))
    buf294 = empty((48, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf292, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf293, (48, 128, 49), (6272, 49, 1), 0), out=buf294)
    buf295 = buf270; del buf270  # reuse
    buf296 = reinterpret_tensor(buf294, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf294  # reuse
    buf297 = empty_strided((8, 6, 49, 1), (294, 49, 1, 2352), device='cpu', dtype=torch.float32)
    buf298 = empty((8, 6, 49, 49), device='cpu', dtype=torch.float32)
    buf299 = empty((8, 6, 49, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_69(c_void_p(buf296.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()))
    del buf291
    del buf295
    buf300 = empty((48, 49, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_156], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf298, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf299, (48, 49, 128), (6272, 128, 1), 0), out=buf300)
    buf301 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_70(c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    # Source Nodes: [x_158], Original ATen: [aten.convolution]
    buf302 = extern_kernels.convolution(buf301, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf302, (8, 768, 7, 7), (37632, 1, 5376, 768))
    buf303 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf304 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf306 = empty((768, ), device='cpu', dtype=torch.float32)
    buf307 = reinterpret_tensor(buf300, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf300  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_71(c_void_p(buf260.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()))
    del primals_115
    # Source Nodes: [x_161], Original ATen: [aten.convolution]
    buf308 = extern_kernels.convolution(buf307, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf308, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    buf309 = empty_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_72(c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()))
    # Source Nodes: [x_164], Original ATen: [aten.convolution]
    buf310 = extern_kernels.convolution(buf309, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf310, (8, 768, 7, 7), (37632, 1, 5376, 768))
    buf311 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    buf312 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf313 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf315 = empty((768, ), device='cpu', dtype=torch.float32)
    buf316 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cpu', dtype=torch.float32)
    buf317 = reinterpret_tensor(buf316, (8, 768), (768, 1), 0); del buf316  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_mean_view_73(c_void_p(buf317.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf315.data_ptr()))
    del buf260
    del buf311
    del primals_119
    buf318 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_121, buf317, reinterpret_tensor(primals_120, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf318)
    del primals_121
    buf319 = empty_strided((8, 6, 49, 49), (14406, 1, 294, 6), device='cpu', dtype=torch.float32)
    buf320 = empty_strided((8, 6, 49, 49), (14406, 1, 294, 6), device='cpu', dtype=torch.float32)
    buf321 = empty_strided((8, 6, 49, 49), (14406, 1, 294, 6), device='cpu', dtype=torch.float32)
    buf322 = empty_strided((8, 6, 49, 49), (14406, 1, 294, 6), device='cpu', dtype=torch.float32)
    buf323 = empty_strided((8, 6, 196, 196), (230496, 1, 1176, 6), device='cpu', dtype=torch.float32)
    buf324 = empty_strided((8, 6, 196, 196), (230496, 1, 1176, 6), device='cpu', dtype=torch.float32)
    buf325 = empty_strided((8, 6, 196, 196), (230496, 1, 1176, 6), device='cpu', dtype=torch.float32)
    buf326 = empty_strided((8, 6, 196, 196), (230496, 1, 1176, 6), device='cpu', dtype=torch.float32)
    buf331 = reinterpret_tensor(buf14, (32, ), (1, ), 0); del buf14  # reuse
    buf339 = reinterpret_tensor(buf20, (192, ), (1, ), 0); del buf20  # reuse
    buf347 = reinterpret_tensor(buf25, (192, ), (1, ), 0); del buf25  # reuse
    buf355 = reinterpret_tensor(buf35, (192, ), (1, ), 0); del buf35  # reuse
    buf363 = reinterpret_tensor(buf45, (192, ), (1, ), 0); del buf45  # reuse
    buf371 = reinterpret_tensor(buf55, (192, ), (1, ), 0); del buf55  # reuse
    buf379 = reinterpret_tensor(buf66, (192, ), (1, ), 0); del buf66  # reuse
    buf387 = reinterpret_tensor(buf76, (192, ), (1, ), 0); del buf76  # reuse
    buf395 = reinterpret_tensor(buf86, (192, ), (1, ), 0); del buf86  # reuse
    buf403 = reinterpret_tensor(buf98, (384, ), (1, ), 0); del buf98  # reuse
    buf411 = reinterpret_tensor(buf103, (384, ), (1, ), 0); del buf103  # reuse
    buf419 = reinterpret_tensor(buf120, (384, ), (1, ), 0); del buf120  # reuse
    buf427 = reinterpret_tensor(buf128, (384, ), (1, ), 0); del buf128  # reuse
    buf435 = reinterpret_tensor(buf145, (384, ), (1, ), 0); del buf145  # reuse
    buf443 = reinterpret_tensor(buf154, (384, ), (1, ), 0); del buf154  # reuse
    buf451 = reinterpret_tensor(buf171, (384, ), (1, ), 0); del buf171  # reuse
    buf459 = reinterpret_tensor(buf179, (384, ), (1, ), 0); del buf179  # reuse
    buf467 = reinterpret_tensor(buf196, (384, ), (1, ), 0); del buf196  # reuse
    buf475 = reinterpret_tensor(buf206, (768, ), (1, ), 0); del buf206  # reuse
    buf483 = reinterpret_tensor(buf211, (768, ), (1, ), 0); del buf211  # reuse
    buf491 = reinterpret_tensor(buf228, (768, ), (1, ), 0); del buf228  # reuse
    buf499 = reinterpret_tensor(buf236, (768, ), (1, ), 0); del buf236  # reuse
    buf507 = reinterpret_tensor(buf253, (768, ), (1, ), 0); del buf253  # reuse
    buf515 = reinterpret_tensor(buf262, (768, ), (1, ), 0); del buf262  # reuse
    buf523 = reinterpret_tensor(buf279, (768, ), (1, ), 0); del buf279  # reuse
    buf531 = reinterpret_tensor(buf287, (768, ), (1, ), 0); del buf287  # reuse
    buf539 = reinterpret_tensor(buf304, (768, ), (1, ), 0); del buf304  # reuse
    buf547 = reinterpret_tensor(buf313, (768, ), (1, ), 0); del buf313  # reuse
    cpp_fused__native_batch_norm_legit_functional__softmax_add_detach_74(c_void_p(buf331.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(primals_205.data_ptr()))
    del buf112
    del buf113
    del buf137
    del buf138
    del buf163
    del buf164
    del buf188
    del buf189
    del buf220
    del buf221
    del buf245
    del buf246
    del buf271
    del buf272
    del buf296
    del buf297
    del buf331
    del buf339
    del buf347
    del buf355
    del buf363
    del buf371
    del buf379
    del buf387
    del buf395
    del buf403
    del buf411
    del buf419
    del buf427
    del buf435
    del buf443
    del buf451
    del buf459
    del buf467
    del buf475
    del buf483
    del buf491
    del buf499
    del buf507
    del buf515
    del buf523
    del buf531
    del buf539
    del buf547
    del primals_122
    del primals_123
    del primals_124
    del primals_125
    del primals_126
    del primals_127
    del primals_128
    del primals_129
    del primals_130
    del primals_131
    del primals_132
    del primals_133
    del primals_134
    del primals_135
    del primals_136
    del primals_137
    del primals_138
    del primals_139
    del primals_140
    del primals_141
    del primals_142
    del primals_143
    del primals_144
    del primals_145
    del primals_146
    del primals_147
    del primals_148
    del primals_149
    del primals_150
    del primals_151
    del primals_152
    del primals_153
    del primals_154
    del primals_155
    del primals_156
    del primals_157
    del primals_158
    del primals_159
    del primals_160
    del primals_161
    del primals_162
    del primals_163
    del primals_164
    del primals_165
    del primals_166
    del primals_167
    del primals_168
    del primals_169
    del primals_170
    del primals_171
    del primals_172
    del primals_173
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
    return (buf318, buf0, primals_5, buf1, primals_9, primals_11, primals_13, buf2, primals_15, primals_16, primals_18, buf3, primals_20, primals_21, primals_23, buf4, primals_25, primals_26, primals_28, buf5, primals_30, primals_31, primals_33, buf6, primals_35, primals_36, primals_38, buf7, primals_40, primals_41, primals_43, buf8, primals_45, buf9, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_80, primals_81, buf10, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_118, buf11, buf12, buf16, buf17, buf18, buf22, buf23, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf88, buf89, buf90, buf91, buf92, buf93, buf95, buf96, buf100, buf101, buf105, buf106, buf117, buf118, buf122, buf123, buf124, buf125, buf126, buf130, buf131, buf142, buf143, buf147, buf148, buf149, buf150, buf151, buf156, buf157, buf168, buf169, buf173, buf174, buf175, buf176, buf177, buf181, buf182, buf193, buf194, buf198, buf199, buf200, buf201, buf203, buf204, buf208, buf209, buf213, buf214, buf225, buf226, buf230, buf231, buf232, buf233, buf234, buf238, buf239, buf250, buf251, buf255, buf256, buf257, buf258, buf259, buf264, buf265, buf276, buf277, buf281, buf282, buf283, buf284, buf285, buf289, buf290, buf301, buf302, buf306, buf307, buf308, buf309, buf310, buf315, buf317, reinterpret_tensor(primals_120, (1000, 768), (768, 1), 0), reinterpret_tensor(buf312, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf303, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf298, (48, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf299, (48, 128, 49), (6272, 1, 128), 0), buf319, reinterpret_tensor(buf292, (48, 128, 49), (6272, 1, 128), 0), reinterpret_tensor(buf293, (48, 49, 128), (6272, 1, 49), 0), reinterpret_tensor(buf286, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf278, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf273, (48, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf274, (48, 128, 49), (6272, 1, 128), 0), buf320, reinterpret_tensor(buf267, (48, 128, 49), (6272, 1, 128), 0), reinterpret_tensor(buf268, (48, 49, 128), (6272, 1, 49), 0), reinterpret_tensor(buf261, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf252, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf247, (48, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf248, (48, 128, 49), (6272, 1, 128), 0), buf321, reinterpret_tensor(buf241, (48, 128, 49), (6272, 1, 128), 0), reinterpret_tensor(buf242, (48, 49, 128), (6272, 1, 49), 0), reinterpret_tensor(buf235, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf227, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf222, (48, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf223, (48, 128, 49), (6272, 1, 128), 0), buf322, reinterpret_tensor(buf216, (48, 128, 49), (6272, 1, 128), 0), reinterpret_tensor(buf217, (48, 49, 128), (6272, 1, 49), 0), reinterpret_tensor(buf210, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf205, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf195, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf190, (48, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf191, (48, 64, 196), (12544, 1, 64), 0), buf323, reinterpret_tensor(buf184, (48, 64, 196), (12544, 1, 64), 0), reinterpret_tensor(buf185, (48, 196, 64), (12544, 1, 196), 0), reinterpret_tensor(buf178, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf170, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf165, (48, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf166, (48, 64, 196), (12544, 1, 64), 0), buf324, reinterpret_tensor(buf159, (48, 64, 196), (12544, 1, 64), 0), reinterpret_tensor(buf160, (48, 196, 64), (12544, 1, 196), 0), reinterpret_tensor(buf153, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf144, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf139, (48, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf140, (48, 64, 196), (12544, 1, 64), 0), buf325, reinterpret_tensor(buf133, (48, 64, 196), (12544, 1, 64), 0), reinterpret_tensor(buf134, (48, 196, 64), (12544, 1, 196), 0), reinterpret_tensor(buf127, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf119, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf114, (48, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf115, (48, 64, 196), (12544, 1, 64), 0), buf326, reinterpret_tensor(buf108, (48, 64, 196), (12544, 1, 64), 0), reinterpret_tensor(buf109, (48, 196, 64), (12544, 1, 196), 0), reinterpret_tensor(buf102, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf97, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf85, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf75, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf65, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf54, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf44, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf34, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf24, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf19, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf13, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 192, 28, 28), (150528, 784, 28, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, 384, 14, 14), (75264, 196, 14, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1, 768, 7, 7), (37632, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((192, 32, 4, 4), (512, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, 384, 2, 2), (1536, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_125 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_128 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_131 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_134 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_137 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_140 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_143 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_146 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_149 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_152 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_155 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_158 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_161 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_164 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_167 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_170 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_173 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_176 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_179 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_182 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_185 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_188 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_191 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_194 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_197 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_200 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_203 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_206 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('visformer_small', benchmark_compiled_module)
