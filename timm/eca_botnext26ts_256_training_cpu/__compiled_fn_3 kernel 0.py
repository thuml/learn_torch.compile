
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (216L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (24L*x2) + (216L*x0)), static_cast<long>(24L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (216L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (24L*x2) + (216L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr6 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
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
                        auto tmp0 = in_ptr8[static_cast<long>(x2 + (65536L*x1) + (196608L*x0))];
                        out_ptr8[static_cast<long>(x1 + (3L*x2) + (196608L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_2 = async_compile.cpp('''
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_max_pool2d_with_indices_silu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       long* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(128);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr4[static_cast<long>((-8256L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
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
                                auto tmp20 = out_ptr4[static_cast<long>((-8192L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
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
                                auto tmp29 = out_ptr4[static_cast<long>((-8128L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
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
                                auto tmp38 = out_ptr4[static_cast<long>((-64L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : -std::numeric_limits<decltype(tmp37())>::infinity();
                            auto tmp40 = max_propagate_nan(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = out_ptr4[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = max_propagate_nan(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = out_ptr4[static_cast<long>(64L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
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
                                auto tmp57 = out_ptr4[static_cast<long>(8128L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : -std::numeric_limits<decltype(tmp56())>::infinity();
                            auto tmp59 = max_propagate_nan(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = out_ptr4[static_cast<long>(8192L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : -std::numeric_limits<decltype(tmp61())>::infinity();
                            auto tmp64 = max_propagate_nan(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = out_ptr4[static_cast<long>(8256L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : -std::numeric_limits<decltype(tmp66())>::infinity();
                            auto tmp69 = max_propagate_nan(tmp68, tmp64);
                            auto tmp70 = [&]
                            {
                                auto tmp71 = out_ptr4[static_cast<long>((-8256L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp10 ? tmp70() : -std::numeric_limits<decltype(tmp70())>::infinity();
                            auto tmp73 = [&]
                            {
                                auto tmp74 = out_ptr4[static_cast<long>((-8192L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp74;
                            }
                            ;
                            auto tmp75 = tmp18 ? tmp73() : -std::numeric_limits<decltype(tmp73())>::infinity();
                            auto tmp76 = tmp75 > tmp72;
                            auto tmp77 = c10::convert<long>((-128L) + (2L*x2) + (256L*x1));
                            auto tmp78 = c10::convert<long>((-129L) + (2L*x2) + (256L*x1));
                            auto tmp79 = tmp76 ? tmp77 : tmp78;
                            auto tmp80 = max_propagate_nan(tmp75, tmp72);
                            auto tmp81 = [&]
                            {
                                auto tmp82 = out_ptr4[static_cast<long>((-8128L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp82;
                            }
                            ;
                            auto tmp83 = tmp27 ? tmp81() : -std::numeric_limits<decltype(tmp81())>::infinity();
                            auto tmp84 = tmp83 > tmp80;
                            auto tmp85 = c10::convert<long>((-127L) + (2L*x2) + (256L*x1));
                            auto tmp86 = tmp84 ? tmp85 : tmp79;
                            auto tmp87 = max_propagate_nan(tmp83, tmp80);
                            auto tmp88 = [&]
                            {
                                auto tmp89 = out_ptr4[static_cast<long>((-64L) + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp89;
                            }
                            ;
                            auto tmp90 = tmp36 ? tmp88() : -std::numeric_limits<decltype(tmp88())>::infinity();
                            auto tmp91 = tmp90 > tmp87;
                            auto tmp92 = c10::convert<long>((-1L) + (2L*x2) + (256L*x1));
                            auto tmp93 = tmp91 ? tmp92 : tmp86;
                            auto tmp94 = max_propagate_nan(tmp90, tmp87);
                            auto tmp95 = [&]
                            {
                                auto tmp96 = out_ptr4[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp96;
                            }
                            ;
                            auto tmp97 = tmp41 ? tmp95() : -std::numeric_limits<decltype(tmp95())>::infinity();
                            auto tmp98 = tmp97 > tmp94;
                            auto tmp99 = c10::convert<long>((2L*x2) + (256L*x1));
                            auto tmp100 = tmp98 ? tmp99 : tmp93;
                            auto tmp101 = max_propagate_nan(tmp97, tmp94);
                            auto tmp102 = [&]
                            {
                                auto tmp103 = out_ptr4[static_cast<long>(64L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp46 ? tmp102() : -std::numeric_limits<decltype(tmp102())>::infinity();
                            auto tmp105 = tmp104 > tmp101;
                            auto tmp106 = c10::convert<long>(1L + (2L*x2) + (256L*x1));
                            auto tmp107 = tmp105 ? tmp106 : tmp100;
                            auto tmp108 = max_propagate_nan(tmp104, tmp101);
                            auto tmp109 = [&]
                            {
                                auto tmp110 = out_ptr4[static_cast<long>(8128L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp110;
                            }
                            ;
                            auto tmp111 = tmp55 ? tmp109() : -std::numeric_limits<decltype(tmp109())>::infinity();
                            auto tmp112 = tmp111 > tmp108;
                            auto tmp113 = c10::convert<long>(127L + (2L*x2) + (256L*x1));
                            auto tmp114 = tmp112 ? tmp113 : tmp107;
                            auto tmp115 = max_propagate_nan(tmp111, tmp108);
                            auto tmp116 = [&]
                            {
                                auto tmp117 = out_ptr4[static_cast<long>(8192L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp117;
                            }
                            ;
                            auto tmp118 = tmp60 ? tmp116() : -std::numeric_limits<decltype(tmp116())>::infinity();
                            auto tmp119 = tmp118 > tmp115;
                            auto tmp120 = c10::convert<long>(128L + (2L*x2) + (256L*x1));
                            auto tmp121 = tmp119 ? tmp120 : tmp114;
                            auto tmp122 = max_propagate_nan(tmp118, tmp115);
                            auto tmp123 = [&]
                            {
                                auto tmp124 = out_ptr4[static_cast<long>(8256L + x3 + (128L*x2) + (16384L*x1) + (1048576L*x0))];
                                return tmp124;
                            }
                            ;
                            auto tmp125 = tmp65 ? tmp123() : -std::numeric_limits<decltype(tmp123())>::infinity();
                            auto tmp126 = tmp125 > tmp122;
                            auto tmp127 = c10::convert<long>(129L + (2L*x2) + (256L*x1));
                            auto tmp128 = tmp126 ? tmp127 : tmp121;
                            auto tmp129 = max_propagate_nan(tmp125, tmp122);
                            out_ptr5[static_cast<long>(x3 + (64L*x2) + (4096L*x1) + (262144L*x0))] = tmp69;
                            out_ptr6[static_cast<long>(x3 + (64L*x2) + (4096L*x1) + (262144L*x0))] = tmp128;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_silu_view_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4096L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (64L*x2) + (262144L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_silu_8 = async_compile.cpp('''
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
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp28.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_silu_view_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4096L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (64L*x2) + (262144L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_silu_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_13 = async_compile.cpp('''
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_silu_view_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (128L*x2) + (131072L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_silu_17 = async_compile.cpp('''
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
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp28.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_silu_view_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (128L*x2) + (131072L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_silu_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_silu_view_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (256L*x2) + (65536L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_silu_26 = async_compile.cpp('''
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
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp28.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_28 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (98304L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (16384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(64L + x1 + (384L*x2) + (98304L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (16384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(c10::div_floor_integer(((16L*(static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))) + (static_cast<long>(x0) % static_cast<long>(16L))), 16L)) % static_cast<long>(16L))) + (256L*(static_cast<long>(c10::div_floor_integer(((16L*(static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))) + (256L*x1) + (256L*x1_inner) + (4096L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(16L))), 256L)) % static_cast<long>(512L))) + (static_cast<long>(x0) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(c10::div_floor_integer(((16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))), 16L)) % static_cast<long>(16L))) + (256L*(static_cast<long>(c10::div_floor_integer(((16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (256L*x1) + (256L*x1_inner) + (4096L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))), 256L)) % static_cast<long>(512L))) + (static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_mul_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (65536L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>(15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)));
                            auto tmp4 = static_cast<long>(512);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)))) % static_cast<long>(32L));
                                auto tmp8 = static_cast<long>(31);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr1[static_cast<long>((31L*(c10::div_floor_integer((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L))), 32L))) + (496L*(static_cast<long>(x1) % static_cast<long>(16L))) + (7936L*x0) + (static_cast<long>((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)))) % static_cast<long>(32L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = c10::convert<long>(15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)));
                            auto tmp15 = tmp14 < tmp4;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(static_cast<long>((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)))) % static_cast<long>(32L));
                                auto tmp18 = static_cast<long>(31);
                                auto tmp19 = tmp17 < tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr2[static_cast<long>((31L*(static_cast<long>(c10::div_floor_integer((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L))), 32L)) % static_cast<long>(16L))) + (496L*(c10::div_floor_integer(x1, 16L))) + (7936L*x0) + (static_cast<long>((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)))) % static_cast<long>(32L)))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                            auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (256L*x1) + (65536L*x0))];
                        auto tmp26 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = c10::convert<long>(15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)));
                        auto tmp4 = static_cast<long>(512);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = c10::convert<long>(static_cast<long>((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)))) % static_cast<long>(32L));
                            auto tmp8 = static_cast<long>(31);
                            auto tmp9 = tmp7 < tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = in_ptr1[static_cast<long>((31L*(c10::div_floor_integer((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L))), 32L))) + (496L*(static_cast<long>(x1) % static_cast<long>(16L))) + (7936L*x0) + (static_cast<long>((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)))) % static_cast<long>(32L)))];
                                return tmp11;
                            }
                            ;
                            auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp14 = c10::convert<long>(15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)));
                        auto tmp15 = tmp14 < tmp4;
                        auto tmp16 = [&]
                        {
                            auto tmp17 = c10::convert<long>(static_cast<long>((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)))) % static_cast<long>(32L));
                            auto tmp18 = static_cast<long>(31);
                            auto tmp19 = tmp17 < tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr2[static_cast<long>((31L*(static_cast<long>(c10::div_floor_integer((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L))), 32L)) % static_cast<long>(16L))) + (496L*(c10::div_floor_integer(x1, 16L))) + (7936L*x0) + (static_cast<long>((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)))) % static_cast<long>(32L)))];
                                return tmp21;
                            }
                            ;
                            auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                        auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                        auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = std::exp(tmp27);
                        in_out_ptr0[static_cast<long>(x2 + (256L*x1) + (65536L*x0))] = tmp28;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(128L + x1 + (384L*x2) + (98304L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (65536L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*x3) + (1024L*x2) + (16384L*(c10::div_floor_integer((x3 + (16L*x2) + (256L*x0) + (256L*x0_inner)), 16384L))) + (65536L*x1) + (static_cast<long>(c10::div_floor_integer((x3 + (16L*x2) + (256L*x0) + (256L*x0_inner)), 256L)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                        }
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (16384L*(c10::div_floor_integer((x1 + (256L*x2) + (256L*x2_inner)), 16384L))) + (65536L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
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
                        tmp15.store(out_ptr3 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_silu_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_35 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (640L*x2) + (163840L*x0)), static_cast<long>(640L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (16384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(64L + x1 + (640L*x2) + (163840L*x0)), static_cast<long>(640L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (16384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(c10::div_floor_integer(((16L*(static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))) + (static_cast<long>(x0) % static_cast<long>(16L))), 16L)) % static_cast<long>(16L))) + (256L*(static_cast<long>(c10::div_floor_integer(((16L*(static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))) + (256L*x1) + (256L*x1_inner) + (4096L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(16L))), 256L)) % static_cast<long>(512L))) + (static_cast<long>(x0) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(c10::div_floor_integer(((16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))), 16L)) % static_cast<long>(16L))) + (256L*(static_cast<long>(c10::div_floor_integer(((16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (256L*x1) + (256L*x1_inner) + (4096L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))), 256L)) % static_cast<long>(512L))) + (static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_mul_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (65536L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>(15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)));
                            auto tmp4 = static_cast<long>(512);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)))) % static_cast<long>(32L));
                                auto tmp8 = static_cast<long>(31);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr1[static_cast<long>((31L*(c10::div_floor_integer((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L))), 32L))) + (496L*(static_cast<long>(x1) % static_cast<long>(16L))) + (7936L*x0) + (static_cast<long>((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)))) % static_cast<long>(32L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = c10::convert<long>(15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)));
                            auto tmp15 = tmp14 < tmp4;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(static_cast<long>((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)))) % static_cast<long>(32L));
                                auto tmp18 = static_cast<long>(31);
                                auto tmp19 = tmp17 < tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr2[static_cast<long>((31L*(static_cast<long>(c10::div_floor_integer((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L))), 32L)) % static_cast<long>(16L))) + (496L*(c10::div_floor_integer(x1, 16L))) + (7936L*x0) + (static_cast<long>((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)))) % static_cast<long>(32L)))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                            auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (256L*x1) + (65536L*x0))];
                        auto tmp26 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = c10::convert<long>(15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)));
                        auto tmp4 = static_cast<long>(512);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = c10::convert<long>(static_cast<long>((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)))) % static_cast<long>(32L));
                            auto tmp8 = static_cast<long>(31);
                            auto tmp9 = tmp7 < tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = in_ptr1[static_cast<long>((31L*(c10::div_floor_integer((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L))), 32L))) + (496L*(static_cast<long>(x1) % static_cast<long>(16L))) + (7936L*x0) + (static_cast<long>((15L + (31L*(c10::div_floor_integer(x1, 16L))) + (c10::div_floor_integer(x2, 16L)))) % static_cast<long>(32L)))];
                                return tmp11;
                            }
                            ;
                            auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp14 = c10::convert<long>(15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)));
                        auto tmp15 = tmp14 < tmp4;
                        auto tmp16 = [&]
                        {
                            auto tmp17 = c10::convert<long>(static_cast<long>((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)))) % static_cast<long>(32L));
                            auto tmp18 = static_cast<long>(31);
                            auto tmp19 = tmp17 < tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr2[static_cast<long>((31L*(static_cast<long>(c10::div_floor_integer((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L))), 32L)) % static_cast<long>(16L))) + (496L*(c10::div_floor_integer(x1, 16L))) + (7936L*x0) + (static_cast<long>((15L + (31L*(static_cast<long>(x1) % static_cast<long>(16L))) + (static_cast<long>(x2) % static_cast<long>(16L)))) % static_cast<long>(32L)))];
                                return tmp21;
                            }
                            ;
                            auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                        auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                        auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = std::exp(tmp27);
                        in_out_ptr0[static_cast<long>(x2 + (256L*x1) + (65536L*x0))] = tmp28;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(128L + x1 + (640L*x2) + (163840L*x0)), static_cast<long>(640L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (131072L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_avg_pool2d_clone_fill_mul_sigmoid_silu_sub_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (32768L*(c10::div_floor_integer((x1 + (256L*x2) + (256L*x2_inner)), 32768L))) + (131072L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (131072L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (16384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(8192L + x2 + (1024L*x1) + (16384L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(8704L + x2 + (1024L*x1) + (16384L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (4096L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp1;
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp1 * tmp7;
                tmp2.store(out_ptr6 + static_cast<long>(x0));
                tmp8.store(out_ptr7 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_41 = async_compile.cpp('''
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
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp28.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp1;
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp1 * tmp7;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
                tmp8.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp1;
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp1 * tmp7;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
                tmp8.store(out_ptr5 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (640L*x2) + (40960L*x0)), static_cast<long>(640L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (4096L*x0)));
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(64L + x1 + (640L*x2) + (40960L*x0)), static_cast<long>(640L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (4096L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((8L*(static_cast<long>(c10::div_floor_integer(((8L*(static_cast<long>(c10::div_floor_integer(x0, 8L)) % static_cast<long>(8L))) + (static_cast<long>(x0) % static_cast<long>(8L))), 8L)) % static_cast<long>(8L))) + (64L*(static_cast<long>(c10::div_floor_integer(((8L*(static_cast<long>(c10::div_floor_integer(x0, 8L)) % static_cast<long>(8L))) + (64L*x1) + (64L*x1_inner) + (1024L*(c10::div_floor_integer(x0, 64L))) + (static_cast<long>(x0) % static_cast<long>(8L))), 64L)) % static_cast<long>(512L))) + (static_cast<long>(x0) % static_cast<long>(8L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((8L*(static_cast<long>(c10::div_floor_integer(((8L*(static_cast<long>(x0) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(x0, 8L)) % static_cast<long>(8L))), 8L)) % static_cast<long>(8L))) + (64L*(static_cast<long>(c10::div_floor_integer(((8L*(static_cast<long>(x0) % static_cast<long>(8L))) + (64L*x1) + (64L*x1_inner) + (1024L*(c10::div_floor_integer(x0, 64L))) + (static_cast<long>(c10::div_floor_integer(x0, 8L)) % static_cast<long>(8L))), 64L)) % static_cast<long>(512L))) + (static_cast<long>(c10::div_floor_integer(x0, 8L)) % static_cast<long>(8L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_mul_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (4096L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>(7L + (15L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 8L)));
                            auto tmp4 = static_cast<long>(128);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>((7L + (15L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 8L)))) % static_cast<long>(16L));
                                auto tmp8 = static_cast<long>(15);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr1[static_cast<long>((15L*(c10::div_floor_integer((7L + (15L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 8L))), 16L))) + (120L*(static_cast<long>(x1) % static_cast<long>(8L))) + (960L*x0) + (static_cast<long>((7L + (15L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 8L)))) % static_cast<long>(16L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = c10::convert<long>(7L + (15L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(8L)));
                            auto tmp15 = tmp14 < tmp4;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(static_cast<long>((7L + (15L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(8L)))) % static_cast<long>(16L));
                                auto tmp18 = static_cast<long>(15);
                                auto tmp19 = tmp17 < tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr2[static_cast<long>((15L*(static_cast<long>(c10::div_floor_integer((7L + (15L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(8L))), 16L)) % static_cast<long>(8L))) + (120L*(c10::div_floor_integer(x1, 8L))) + (960L*x0) + (static_cast<long>((7L + (15L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(8L)))) % static_cast<long>(16L)))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                            auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (64L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (64L*x1) + (4096L*x0))];
                        auto tmp26 = out_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = c10::convert<long>(7L + (15L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 8L)));
                        auto tmp4 = static_cast<long>(128);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = c10::convert<long>(static_cast<long>((7L + (15L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 8L)))) % static_cast<long>(16L));
                            auto tmp8 = static_cast<long>(15);
                            auto tmp9 = tmp7 < tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = in_ptr1[static_cast<long>((15L*(c10::div_floor_integer((7L + (15L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 8L))), 16L))) + (120L*(static_cast<long>(x1) % static_cast<long>(8L))) + (960L*x0) + (static_cast<long>((7L + (15L*(c10::div_floor_integer(x1, 8L))) + (c10::div_floor_integer(x2, 8L)))) % static_cast<long>(16L)))];
                                return tmp11;
                            }
                            ;
                            auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp14 = c10::convert<long>(7L + (15L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(8L)));
                        auto tmp15 = tmp14 < tmp4;
                        auto tmp16 = [&]
                        {
                            auto tmp17 = c10::convert<long>(static_cast<long>((7L + (15L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(8L)))) % static_cast<long>(16L));
                            auto tmp18 = static_cast<long>(15);
                            auto tmp19 = tmp17 < tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr2[static_cast<long>((15L*(static_cast<long>(c10::div_floor_integer((7L + (15L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(8L))), 16L)) % static_cast<long>(8L))) + (120L*(c10::div_floor_integer(x1, 8L))) + (960L*x0) + (static_cast<long>((7L + (15L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>(x2) % static_cast<long>(8L)))) % static_cast<long>(16L)))];
                                return tmp21;
                            }
                            ;
                            auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                        auto tmp24 = decltype(tmp13)(tmp13 + tmp23);
                        auto tmp25 = decltype(tmp2)(tmp2 + tmp24);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = std::exp(tmp27);
                        in_out_ptr0[static_cast<long>(x2 + (64L*x1) + (4096L*x0))] = tmp28;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(128L + x1 + (640L*x2) + (40960L*x0)), static_cast<long>(640L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*x3) + (1024L*x2) + (8192L*(c10::div_floor_integer((x3 + (8L*x2) + (64L*x0) + (64L*x0_inner)), 8192L))) + (32768L*x1) + (static_cast<long>(c10::div_floor_integer((x3 + (8L*x2) + (64L*x0) + (64L*x0_inner)), 64L)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                        }
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (8192L*(c10::div_floor_integer((x1 + (64L*x2) + (64L*x2_inner)), 8192L))) + (32768L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
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
                        tmp15.store(out_ptr3 + static_cast<long>(x2 + (512L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp1;
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp1 * tmp7;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
                tmp8.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_mean_silu_view_48 = async_compile.cpp('''
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
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (2048L*x2) + (131072L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_49 = async_compile.cpp('''
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const long* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const long* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const long* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const long* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const long* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const long* in_ptr24,
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const long* in_ptr28,
                       const float* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const long* in_ptr32,
                       const float* in_ptr33,
                       const float* in_ptr34,
                       const float* in_ptr35,
                       const long* in_ptr36,
                       const float* in_ptr37,
                       const float* in_ptr38,
                       const float* in_ptr39,
                       const long* in_ptr40,
                       const float* in_ptr41,
                       const float* in_ptr42,
                       const float* in_ptr43,
                       const long* in_ptr44,
                       const float* in_ptr45,
                       const float* in_ptr46,
                       const float* in_ptr47,
                       const long* in_ptr48,
                       const float* in_ptr49,
                       const float* in_ptr50,
                       const float* in_ptr51,
                       const long* in_ptr52,
                       const float* in_ptr53,
                       const float* in_ptr54,
                       const float* in_ptr55,
                       const long* in_ptr56,
                       const float* in_ptr57,
                       const float* in_ptr58,
                       const float* in_ptr59,
                       const long* in_ptr60,
                       const float* in_ptr61,
                       const float* in_ptr62,
                       const float* in_ptr63,
                       const long* in_ptr64,
                       const float* in_ptr65,
                       const float* in_ptr66,
                       const float* in_ptr67,
                       const long* in_ptr68,
                       const float* in_ptr69,
                       const float* in_ptr70,
                       const float* in_ptr71,
                       const long* in_ptr72,
                       const float* in_ptr73,
                       const float* in_ptr74,
                       const float* in_ptr75,
                       const long* in_ptr76,
                       const float* in_ptr77,
                       const float* in_ptr78,
                       const float* in_ptr79,
                       const long* in_ptr80,
                       const float* in_ptr81,
                       const float* in_ptr82,
                       const float* in_ptr83,
                       const long* in_ptr84,
                       const float* in_ptr85,
                       const float* in_ptr86,
                       const float* in_ptr87,
                       const long* in_ptr88,
                       const float* in_ptr89,
                       const float* in_ptr90,
                       const float* in_ptr91,
                       const long* in_ptr92,
                       const float* in_ptr93,
                       const float* in_ptr94,
                       const float* in_ptr95,
                       const long* in_ptr96,
                       const float* in_ptr97,
                       const float* in_ptr98,
                       const float* in_ptr99,
                       const long* in_ptr100,
                       const float* in_ptr101,
                       const float* in_ptr102,
                       const float* in_ptr103,
                       const long* in_ptr104,
                       const float* in_ptr105,
                       const float* in_ptr106,
                       const float* in_ptr107,
                       const long* in_ptr108,
                       const float* in_ptr109,
                       const float* in_ptr110,
                       const float* in_ptr111,
                       const long* in_ptr112,
                       const float* in_ptr113,
                       const float* in_ptr114,
                       const float* in_ptr115,
                       const long* in_ptr116,
                       const float* in_ptr117,
                       const float* in_ptr118,
                       const float* in_ptr119,
                       const long* in_ptr120,
                       const float* in_ptr121,
                       const float* in_ptr122,
                       const float* in_ptr123,
                       long* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr4,
                       long* out_ptr6,
                       float* out_ptr8,
                       float* out_ptr9,
                       long* out_ptr11,
                       float* out_ptr13,
                       float* out_ptr14,
                       long* out_ptr16,
                       float* out_ptr18,
                       float* out_ptr19,
                       long* out_ptr21,
                       float* out_ptr23,
                       float* out_ptr24,
                       long* out_ptr26,
                       float* out_ptr28,
                       float* out_ptr29,
                       long* out_ptr31,
                       float* out_ptr33,
                       float* out_ptr34,
                       long* out_ptr36,
                       float* out_ptr38,
                       float* out_ptr39,
                       long* out_ptr41,
                       float* out_ptr43,
                       float* out_ptr44,
                       long* out_ptr46,
                       float* out_ptr48,
                       float* out_ptr49,
                       long* out_ptr51,
                       float* out_ptr53,
                       float* out_ptr54,
                       long* out_ptr56,
                       float* out_ptr58,
                       float* out_ptr59,
                       long* out_ptr61,
                       float* out_ptr63,
                       float* out_ptr64,
                       long* out_ptr66,
                       float* out_ptr68,
                       float* out_ptr69,
                       long* out_ptr71,
                       float* out_ptr73,
                       float* out_ptr74,
                       long* out_ptr76,
                       float* out_ptr78,
                       float* out_ptr79,
                       long* out_ptr81,
                       float* out_ptr83,
                       float* out_ptr84,
                       long* out_ptr86,
                       float* out_ptr88,
                       float* out_ptr89,
                       long* out_ptr91,
                       float* out_ptr93,
                       float* out_ptr94,
                       long* out_ptr96,
                       float* out_ptr98,
                       float* out_ptr99,
                       long* out_ptr101,
                       float* out_ptr103,
                       float* out_ptr104,
                       long* out_ptr106,
                       float* out_ptr108,
                       float* out_ptr109,
                       long* out_ptr111,
                       float* out_ptr113,
                       float* out_ptr114,
                       long* out_ptr116,
                       float* out_ptr118,
                       float* out_ptr119,
                       long* out_ptr121,
                       float* out_ptr123,
                       float* out_ptr124,
                       long* out_ptr126,
                       float* out_ptr128,
                       float* out_ptr129,
                       long* out_ptr131,
                       float* out_ptr133,
                       float* out_ptr134,
                       long* out_ptr136,
                       float* out_ptr138,
                       float* out_ptr139,
                       long* out_ptr141,
                       float* out_ptr143,
                       float* out_ptr144,
                       long* out_ptr146,
                       float* out_ptr148,
                       float* out_ptr149,
                       long* out_ptr151,
                       float* out_ptr153,
                       float* out_ptr154)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr3 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr6 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr8 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr13 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr14 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr15 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr16 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr17 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr0[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr1[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr4[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr6[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr8 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr9 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr9 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr8[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr11[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr13 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr14 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr14 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr12[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr16[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr18 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr19 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr19 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr16[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr21[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr23 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr24 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr24 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr20[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr26[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr28 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr28 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr29 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr29 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr24[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr31[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr33 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr33 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr34 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr34 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr28[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr36[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr38 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr38 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr39 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr39 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr32[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr41[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr33 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr43 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr43 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr44 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr44 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr36[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr46[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr37 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr48 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr48 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr49 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr49 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr40[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr51[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr41 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr53 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr53 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr54 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr54 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr44[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr56[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr45 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr58 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr58 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr59 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr59 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr48[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr61[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr49 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr63 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr63 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr64 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr64 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr52[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr66[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr53 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr68 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr68 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr69 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr69 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr56[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr71[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr57 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr73 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr73 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr74 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr74 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr60[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr76[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr61 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr78 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr78 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr79 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr79 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr64[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr81[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr65 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr83 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr83 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr84 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr84 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr68[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr86[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr69 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr88 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr88 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr89 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr89 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr72[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr91[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr73 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr93 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr93 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr94 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr94 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr76[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr96[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr77 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr98 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr98 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr99 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr99 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr80[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr101[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr81 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr103 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr103 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr104 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr104 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr84[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr106[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr85 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr108 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr108 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr109 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr109 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr88[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr111[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr89 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr113 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr113 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr114 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr114 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr92[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr116[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr93 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr118 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr118 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr119 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr119 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr96[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr121[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr97 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr123 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr123 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr124 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr124 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr100[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr126[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr101 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr128 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr128 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr129 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr129 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr104[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr131[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr105 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr133 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr133 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr134 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr134 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr108[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr136[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr109 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr138 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr138 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr139 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr139 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr112[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr141[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr113 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr143 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr143 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr144 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr144 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr116[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr146[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr117 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr148 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr148 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr149 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr149 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr120[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr151[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr121 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr153 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr153 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr154 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr154 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200 = args
    args.clear()
    assert_size_stride(primals_1, (24, ), (1, ))
    assert_size_stride(primals_2, (24, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (1024, ), (1, ))
    assert_size_stride(primals_40, (1024, ), (1, ))
    assert_size_stride(primals_41, (1024, ), (1, ))
    assert_size_stride(primals_42, (1024, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (31, 16), (16, 1))
    assert_size_stride(primals_46, (31, 16), (16, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_50, (1024, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (31, 16), (16, 1))
    assert_size_stride(primals_54, (31, 16), (16, 1))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (2048, ), (1, ))
    assert_size_stride(primals_58, (2048, ), (1, ))
    assert_size_stride(primals_59, (2048, ), (1, ))
    assert_size_stride(primals_60, (2048, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (15, 16), (16, 1))
    assert_size_stride(primals_64, (15, 16), (16, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (2048, ), (1, ))
    assert_size_stride(primals_68, (2048, ), (1, ))
    assert_size_stride(primals_69, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_70, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_71, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_72, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_74, (1, 1, 3), (3, 3, 1))
    assert_size_stride(primals_75, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_76, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_77, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_78, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_79, (1, 1, 3), (3, 3, 1))
    assert_size_stride(primals_80, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_81, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_82, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_83, (1, 1, 5), (5, 5, 1))
    assert_size_stride(primals_84, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_85, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_86, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_87, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_88, (1, 1, 5), (5, 5, 1))
    assert_size_stride(primals_89, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_90, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_91, (256, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_92, (1, 1, 5), (5, 5, 1))
    assert_size_stride(primals_93, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_94, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_95, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_96, (384, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_97, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_98, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_99, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_100, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_102, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_103, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_104, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_105, (1000, 2048), (2048, 1))
    assert_size_stride(primals_106, (1000, ), (1, ))
    assert_size_stride(primals_107, (), ())
    assert_size_stride(primals_108, (24, ), (1, ))
    assert_size_stride(primals_109, (24, ), (1, ))
    assert_size_stride(primals_110, (), ())
    assert_size_stride(primals_111, (32, ), (1, ))
    assert_size_stride(primals_112, (32, ), (1, ))
    assert_size_stride(primals_113, (), ())
    assert_size_stride(primals_114, (64, ), (1, ))
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_116, (), ())
    assert_size_stride(primals_117, (64, ), (1, ))
    assert_size_stride(primals_118, (64, ), (1, ))
    assert_size_stride(primals_119, (), ())
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (), ())
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (), ())
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (256, ), (1, ))
    assert_size_stride(primals_128, (), ())
    assert_size_stride(primals_129, (64, ), (1, ))
    assert_size_stride(primals_130, (64, ), (1, ))
    assert_size_stride(primals_131, (), ())
    assert_size_stride(primals_132, (64, ), (1, ))
    assert_size_stride(primals_133, (64, ), (1, ))
    assert_size_stride(primals_134, (), ())
    assert_size_stride(primals_135, (256, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (), ())
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (), ())
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (), ())
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (512, ), (1, ))
    assert_size_stride(primals_146, (), ())
    assert_size_stride(primals_147, (512, ), (1, ))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (), ())
    assert_size_stride(primals_150, (128, ), (1, ))
    assert_size_stride(primals_151, (128, ), (1, ))
    assert_size_stride(primals_152, (), ())
    assert_size_stride(primals_153, (128, ), (1, ))
    assert_size_stride(primals_154, (128, ), (1, ))
    assert_size_stride(primals_155, (), ())
    assert_size_stride(primals_156, (512, ), (1, ))
    assert_size_stride(primals_157, (512, ), (1, ))
    assert_size_stride(primals_158, (), ())
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_161, (), ())
    assert_size_stride(primals_162, (256, ), (1, ))
    assert_size_stride(primals_163, (256, ), (1, ))
    assert_size_stride(primals_164, (), ())
    assert_size_stride(primals_165, (1024, ), (1, ))
    assert_size_stride(primals_166, (1024, ), (1, ))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (1024, ), (1, ))
    assert_size_stride(primals_169, (1024, ), (1, ))
    assert_size_stride(primals_170, (), ())
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (1024, ), (1, ))
    assert_size_stride(primals_178, (1024, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (512, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (512, ), (1, ))
    assert_size_stride(primals_184, (512, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (2048, ), (1, ))
    assert_size_stride(primals_187, (2048, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (2048, ), (1, ))
    assert_size_stride(primals_190, (2048, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (512, ), (1, ))
    assert_size_stride(primals_193, (512, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (512, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (2048, ), (1, ))
    assert_size_stride(primals_199, (2048, ), (1, ))
    assert_size_stride(primals_200, (8, 3, 256, 256), (196608, 65536, 256, 1))
    buf0 = empty_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 24, 3, 3), (216, 1, 72, 24), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((256, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_69.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del primals_200
    del primals_69
    del primals_70
    del primals_71
    del primals_73
    del primals_78
    del primals_82
    del primals_87
    del primals_91
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf9 = extern_kernels.convolution(buf8, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf9, (8, 24, 128, 128), (393216, 1, 3072, 24))
    buf10 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf13 = empty((24, ), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_1(c_void_p(buf9.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    del primals_2
    # Source Nodes: [x_5], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf15, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf16, (8, 32, 128, 128), (524288, 1, 4096, 32))
    buf17 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf20 = empty((32, ), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_2(c_void_p(buf16.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del primals_4
    # Source Nodes: [x_10], Original ATen: [aten.convolution]
    buf23 = extern_kernels.convolution(buf22, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    buf24 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf27 = empty((64, ), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    buf29 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    buf30 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_functional_max_pool2d_with_indices_silu_3(c_void_p(buf23.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del primals_6
    # Source Nodes: [x_16], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf30, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf32, (8, 64, 64, 64), (262144, 1, 4096, 64))
    buf33 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf36 = empty((64, ), device='cpu', dtype=torch.float32)
    buf37 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    buf38 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_4(c_void_p(buf32.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    del primals_8
    # Source Nodes: [x_22], Original ATen: [aten.convolution]
    buf39 = extern_kernels.convolution(buf38, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
    assert_size_stride(buf39, (8, 64, 64, 64), (262144, 1, 4096, 64))
    buf40 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf43 = empty((64, ), device='cpu', dtype=torch.float32)
    buf44 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    buf45 = empty((8, 64), device='cpu', dtype=torch.float32)
    buf46 = reinterpret_tensor(buf45, (8, 1, 64), (64, 64, 1), 0); del buf45  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_silu_view_5(c_void_p(buf46.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    del primals_10
    # Source Nodes: [y_1], Original ATen: [aten.convolution]
    buf47 = extern_kernels.convolution(buf46, primals_74, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf47, (8, 1, 64), (64, 64, 1))
    buf48 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_6(c_void_p(buf44.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    # Source Nodes: [x_30], Original ATen: [aten.convolution]
    buf49 = extern_kernels.convolution(buf48, primals_75, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    buf50 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf51 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf53 = empty((256, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_7(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()))
    # Source Nodes: [x_38], Original ATen: [aten.convolution]
    buf54 = extern_kernels.convolution(buf30, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf54, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    buf55 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf56 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf58 = empty((256, ), device='cpu', dtype=torch.float32)
    buf59 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf60 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_silu_8(c_void_p(buf54.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    del primals_12
    del primals_14
    # Source Nodes: [x_44], Original ATen: [aten.convolution]
    buf61 = extern_kernels.convolution(buf60, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf61, (8, 64, 64, 64), (262144, 1, 4096, 64))
    buf62 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf63 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf65 = empty((64, ), device='cpu', dtype=torch.float32)
    buf66 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    buf67 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_9(c_void_p(buf61.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    del primals_16
    # Source Nodes: [x_50], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
    assert_size_stride(buf68, (8, 64, 64, 64), (262144, 1, 4096, 64))
    buf69 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf70 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf72 = empty((64, ), device='cpu', dtype=torch.float32)
    buf73 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    buf74 = empty((8, 64), device='cpu', dtype=torch.float32)
    buf75 = reinterpret_tensor(buf74, (8, 1, 64), (64, 64, 1), 0); del buf74  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_silu_view_10(c_void_p(buf75.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_18
    # Source Nodes: [y_4], Original ATen: [aten.convolution]
    buf76 = extern_kernels.convolution(buf75, primals_79, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf76, (8, 1, 64), (64, 64, 1))
    buf77 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_11(c_void_p(buf73.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()))
    # Source Nodes: [x_58], Original ATen: [aten.convolution]
    buf78 = extern_kernels.convolution(buf77, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf78, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    buf79 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf80 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf82 = empty((256, ), device='cpu', dtype=torch.float32)
    buf83 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf84 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_silu_12(c_void_p(buf78.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    del primals_20
    # Source Nodes: [x_67], Original ATen: [aten.convolution]
    buf85 = extern_kernels.convolution(buf84, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf85, (8, 128, 64, 64), (524288, 1, 8192, 128))
    buf86 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf87 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf89 = empty((128, ), device='cpu', dtype=torch.float32)
    buf90 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    buf91 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_13(c_void_p(buf85.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    del primals_22
    # Source Nodes: [x_73], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf92, (8, 128, 32, 32), (131072, 1, 4096, 128))
    buf93 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf94 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf96 = empty((128, ), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    buf98 = empty((8, 128), device='cpu', dtype=torch.float32)
    buf99 = reinterpret_tensor(buf98, (8, 1, 128), (128, 128, 1), 0); del buf98  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_silu_view_14(c_void_p(buf99.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_24
    # Source Nodes: [y_7], Original ATen: [aten.convolution]
    buf100 = extern_kernels.convolution(buf99, primals_83, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf100, (8, 1, 128), (128, 128, 1))
    buf101 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_15(c_void_p(buf97.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    # Source Nodes: [x_81], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, primals_84, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (8, 512, 32, 32), (524288, 1, 16384, 512))
    buf103 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf104 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf106 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_16(c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()))
    # Source Nodes: [x_89], Original ATen: [aten.convolution]
    buf107 = extern_kernels.convolution(buf84, primals_85, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf107, (8, 512, 32, 32), (524288, 1, 16384, 512))
    buf108 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf109 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf111 = empty((512, ), device='cpu', dtype=torch.float32)
    buf112 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    buf113 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_silu_17(c_void_p(buf107.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()))
    del primals_26
    del primals_28
    # Source Nodes: [x_95], Original ATen: [aten.convolution]
    buf114 = extern_kernels.convolution(buf113, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (8, 128, 32, 32), (131072, 1, 4096, 128))
    buf115 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf116 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf118 = empty((128, ), device='cpu', dtype=torch.float32)
    buf119 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    buf120 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_18(c_void_p(buf114.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()))
    del primals_30
    # Source Nodes: [x_101], Original ATen: [aten.convolution]
    buf121 = extern_kernels.convolution(buf120, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf121, (8, 128, 32, 32), (131072, 1, 4096, 128))
    buf122 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf123 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf125 = empty((128, ), device='cpu', dtype=torch.float32)
    buf126 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    buf127 = empty((8, 128), device='cpu', dtype=torch.float32)
    buf128 = reinterpret_tensor(buf127, (8, 1, 128), (128, 128, 1), 0); del buf127  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_silu_view_19(c_void_p(buf128.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    del primals_32
    # Source Nodes: [y_10], Original ATen: [aten.convolution]
    buf129 = extern_kernels.convolution(buf128, primals_88, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf129, (8, 1, 128), (128, 128, 1))
    buf130 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_20(c_void_p(buf126.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    # Source Nodes: [x_109], Original ATen: [aten.convolution]
    buf131 = extern_kernels.convolution(buf130, primals_89, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf131, (8, 512, 32, 32), (524288, 1, 16384, 512))
    buf132 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf133 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf135 = empty((512, ), device='cpu', dtype=torch.float32)
    buf136 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    buf137 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_silu_21(c_void_p(buf131.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()))
    del primals_34
    # Source Nodes: [x_118], Original ATen: [aten.convolution]
    buf138 = extern_kernels.convolution(buf137, primals_90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf138, (8, 256, 32, 32), (262144, 1, 8192, 256))
    buf139 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf140 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf142 = empty((256, ), device='cpu', dtype=torch.float32)
    buf143 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    buf144 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_22(c_void_p(buf138.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    del primals_36
    # Source Nodes: [x_124], Original ATen: [aten.convolution]
    buf145 = extern_kernels.convolution(buf144, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
    assert_size_stride(buf145, (8, 256, 16, 16), (65536, 1, 4096, 256))
    buf146 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf147 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf149 = empty((256, ), device='cpu', dtype=torch.float32)
    buf150 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    buf151 = empty((8, 256), device='cpu', dtype=torch.float32)
    buf152 = reinterpret_tensor(buf151, (8, 1, 256), (256, 256, 1), 0); del buf151  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_silu_view_23(c_void_p(buf152.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    del primals_38
    # Source Nodes: [y_13], Original ATen: [aten.convolution]
    buf153 = extern_kernels.convolution(buf152, primals_92, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf153, (8, 1, 256), (256, 256, 1))
    buf154 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_24(c_void_p(buf150.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    # Source Nodes: [x_132], Original ATen: [aten.convolution]
    buf155 = extern_kernels.convolution(buf154, primals_93, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf155, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    buf156 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf157 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf159 = empty((1024, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_25(c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()))
    # Source Nodes: [x_140], Original ATen: [aten.convolution]
    buf160 = extern_kernels.convolution(buf137, primals_94, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf160, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    buf161 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf162 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf164 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf165 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    buf166 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_silu_26(c_void_p(buf160.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    del primals_40
    del primals_42
    # Source Nodes: [x_146], Original ATen: [aten.convolution]
    buf167 = extern_kernels.convolution(buf166, primals_95, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf167, (8, 256, 16, 16), (65536, 1, 4096, 256))
    buf168 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf169 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf171 = empty((256, ), device='cpu', dtype=torch.float32)
    buf172 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    buf173 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_27(c_void_p(buf167.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del primals_44
    # Source Nodes: [x_153], Original ATen: [aten.convolution]
    buf174 = extern_kernels.convolution(buf173, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf174, (8, 384, 16, 16), (98304, 1, 6144, 384))
    buf175 = empty((8, 64, 16, 16), device='cpu', dtype=torch.float32)
    buf176 = empty((8, 64, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_28(c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    buf177 = empty((32, 256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf175, (32, 256, 16), (4096, 1, 256), 0), reinterpret_tensor(buf176, (32, 16, 256), (4096, 256, 1), 0), out=buf177)
    buf178 = empty((8192, 16), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_29(c_void_p(buf175.data_ptr()), c_void_p(buf178.data_ptr()))
    buf179 = empty((8192, 31), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_154], Original ATen: [aten.mm]
    extern_kernels.mm(buf178, reinterpret_tensor(primals_45, (16, 31), (1, 16), 0), out=buf179)
    buf180 = empty((8192, 16), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_30(c_void_p(buf175.data_ptr()), c_void_p(buf180.data_ptr()))
    buf181 = empty((8192, 31), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_158], Original ATen: [aten.mm]
    extern_kernels.mm(buf180, reinterpret_tensor(primals_46, (16, 31), (1, 16), 0), out=buf181)
    buf182 = empty_strided((32, 256, 1), (256, 1, 8192), device='cpu', dtype=torch.float32)
    buf183 = buf177; del buf177  # reuse
    buf184 = empty_strided((32, 256, 1), (256, 1, 8192), device='cpu', dtype=torch.float32)
    buf185 = buf183; del buf183  # reuse
    buf186 = empty((8, 256, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_mul_31(c_void_p(buf185.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()))
    del buf174
    buf187 = empty((32, 256, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf185, reinterpret_tensor(buf186, (32, 256, 64), (16384, 1, 256), 0), out=buf187)
    buf188 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf189 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf191 = empty((256, ), device='cpu', dtype=torch.float32)
    buf192 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    buf193 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_32(c_void_p(buf187.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    del primals_48
    # Source Nodes: [x_167], Original ATen: [aten.convolution]
    buf194 = extern_kernels.convolution(buf193, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf194, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    buf195 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf196 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf198 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf199 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    buf200 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_silu_33(c_void_p(buf194.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()))
    del primals_50
    # Source Nodes: [x_175], Original ATen: [aten.convolution]
    buf201 = extern_kernels.convolution(buf200, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf201, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf202 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf203 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf205 = empty((512, ), device='cpu', dtype=torch.float32)
    buf206 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    buf207 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_34(c_void_p(buf201.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    del primals_52
    # Source Nodes: [x_182], Original ATen: [aten.convolution]
    buf208 = extern_kernels.convolution(buf207, primals_99, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf208, (8, 640, 16, 16), (163840, 1, 10240, 640))
    buf209 = empty((8, 64, 16, 16), device='cpu', dtype=torch.float32)
    buf210 = empty((8, 64, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_35(c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    buf211 = empty((32, 256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf209, (32, 256, 16), (4096, 1, 256), 0), reinterpret_tensor(buf210, (32, 16, 256), (4096, 256, 1), 0), out=buf211)
    buf212 = empty((8192, 16), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_36(c_void_p(buf209.data_ptr()), c_void_p(buf212.data_ptr()))
    buf213 = buf181; del buf181  # reuse
    # Source Nodes: [x_183], Original ATen: [aten.mm]
    extern_kernels.mm(buf212, reinterpret_tensor(primals_53, (16, 31), (1, 16), 0), out=buf213)
    buf214 = empty((8192, 16), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_37(c_void_p(buf209.data_ptr()), c_void_p(buf214.data_ptr()))
    buf215 = buf179; del buf179  # reuse
    # Source Nodes: [x_187], Original ATen: [aten.mm]
    extern_kernels.mm(buf214, reinterpret_tensor(primals_54, (16, 31), (1, 16), 0), out=buf215)
    buf216 = buf184; del buf184  # reuse
    buf217 = buf211; del buf211  # reuse
    buf218 = buf182; del buf182  # reuse
    buf219 = buf217; del buf217  # reuse
    buf220 = empty((8, 512, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_mul_38(c_void_p(buf219.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()))
    del buf208
    del buf213
    del buf215
    del buf216
    del buf218
    buf221 = empty((32, 256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf219, reinterpret_tensor(buf220, (32, 256, 128), (32768, 1, 256), 0), out=buf221)
    buf222 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    buf223 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    buf224 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf225 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf227 = empty((512, ), device='cpu', dtype=torch.float32)
    buf228 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    buf229 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    buf282 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_avg_pool2d_clone_fill_mul_sigmoid_silu_sub_39(c_void_p(buf221.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf282.data_ptr()))
    del primals_56
    # Source Nodes: [x_196], Original ATen: [aten.convolution]
    buf230 = extern_kernels.convolution(buf229, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf230, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    buf231 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf232 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf234 = empty((2048, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_40(c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()))
    # Source Nodes: [x_203], Original ATen: [aten.convolution]
    buf235 = extern_kernels.convolution(buf200, primals_101, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf235, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    buf236 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf237 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf239 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf240 = reinterpret_tensor(buf221, (8, 2048, 8, 8), (131072, 1, 16384, 2048), 0); del buf221  # reuse
    buf241 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    buf281 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_41(c_void_p(buf235.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf281.data_ptr()))
    del primals_58
    del primals_60
    # Source Nodes: [x_209], Original ATen: [aten.convolution]
    buf242 = extern_kernels.convolution(buf241, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf242, (8, 512, 8, 8), (32768, 1, 4096, 512))
    buf243 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf244 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf246 = empty((512, ), device='cpu', dtype=torch.float32)
    buf247 = buf228; del buf228  # reuse
    buf248 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    buf280 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_42(c_void_p(buf242.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf280.data_ptr()))
    del primals_62
    # Source Nodes: [x_216], Original ATen: [aten.convolution]
    buf249 = extern_kernels.convolution(buf248, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf249, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf250 = empty((8, 64, 8, 8), device='cpu', dtype=torch.float32)
    buf251 = empty((8, 64, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_clone_43(c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    buf252 = empty((32, 64, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf250, (32, 64, 16), (1024, 1, 64), 0), reinterpret_tensor(buf251, (32, 16, 64), (1024, 64, 1), 0), out=buf252)
    buf253 = empty((2048, 16), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_44(c_void_p(buf250.data_ptr()), c_void_p(buf253.data_ptr()))
    buf254 = empty((2048, 15), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_217], Original ATen: [aten.mm]
    extern_kernels.mm(buf253, reinterpret_tensor(primals_63, (16, 15), (1, 16), 0), out=buf254)
    buf255 = empty((2048, 16), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_45(c_void_p(buf250.data_ptr()), c_void_p(buf255.data_ptr()))
    buf256 = empty((2048, 15), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_221], Original ATen: [aten.mm]
    extern_kernels.mm(buf255, reinterpret_tensor(primals_64, (16, 15), (1, 16), 0), out=buf256)
    buf257 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf258 = buf252; del buf252  # reuse
    buf259 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf260 = buf258; del buf258  # reuse
    buf261 = reinterpret_tensor(buf247, (8, 512, 8, 8), (32768, 64, 8, 1), 0); del buf247  # reuse
    cpp_fused__softmax_add_clone_mul_46(c_void_p(buf260.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()))
    del buf249
    del buf254
    del buf256
    buf262 = empty((32, 64, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf260, reinterpret_tensor(buf261, (32, 64, 128), (8192, 1, 64), 0), out=buf262)
    buf263 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf264 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf266 = empty((512, ), device='cpu', dtype=torch.float32)
    buf267 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    buf268 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    buf279 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_47(c_void_p(buf262.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf279.data_ptr()))
    del buf267
    del primals_66
    # Source Nodes: [x_230], Original ATen: [aten.convolution]
    buf269 = extern_kernels.convolution(buf268, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf269, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    buf270 = reinterpret_tensor(buf259, (1, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf259  # reuse
    buf271 = reinterpret_tensor(buf257, (1, 2048, 1, 1), (2048, 1, 2048, 2048), 0); del buf257  # reuse
    buf273 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf274 = buf240; del buf240  # reuse
    buf275 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cpu', dtype=torch.float32)
    buf276 = reinterpret_tensor(buf275, (8, 2048), (2048, 1), 0); del buf275  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_mean_silu_view_48(c_void_p(buf276.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del primals_68
    buf277 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_245], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_106, buf276, reinterpret_tensor(primals_105, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf277)
    del primals_106
    buf278 = buf274; del buf274  # reuse
    buf283 = buf206; del buf206  # reuse
    buf284 = buf199; del buf199  # reuse
    buf285 = buf192; del buf192  # reuse
    buf286 = buf172; del buf172  # reuse
    buf287 = buf165; del buf165  # reuse
    buf288 = buf143; del buf143  # reuse
    buf289 = buf136; del buf136  # reuse
    buf290 = buf119; del buf119  # reuse
    buf291 = buf112; del buf112  # reuse
    buf292 = buf90; del buf90  # reuse
    buf293 = buf83; del buf83  # reuse
    buf294 = buf66; del buf66  # reuse
    buf295 = buf59; del buf59  # reuse
    buf296 = buf37; del buf37  # reuse
    buf297 = buf28; del buf28  # reuse
    buf298 = buf21; del buf21  # reuse
    buf299 = buf14; del buf14  # reuse
    buf306 = reinterpret_tensor(buf11, (24, ), (1, ), 0); del buf11  # reuse
    buf314 = reinterpret_tensor(buf18, (32, ), (1, ), 0); del buf18  # reuse
    buf322 = reinterpret_tensor(buf25, (64, ), (1, ), 0); del buf25  # reuse
    buf330 = reinterpret_tensor(buf34, (64, ), (1, ), 0); del buf34  # reuse
    buf338 = reinterpret_tensor(buf41, (64, ), (1, ), 0); del buf41  # reuse
    buf346 = reinterpret_tensor(buf51, (256, ), (1, ), 0); del buf51  # reuse
    buf354 = reinterpret_tensor(buf56, (256, ), (1, ), 0); del buf56  # reuse
    buf362 = reinterpret_tensor(buf63, (64, ), (1, ), 0); del buf63  # reuse
    buf370 = reinterpret_tensor(buf70, (64, ), (1, ), 0); del buf70  # reuse
    buf378 = reinterpret_tensor(buf80, (256, ), (1, ), 0); del buf80  # reuse
    buf386 = reinterpret_tensor(buf87, (128, ), (1, ), 0); del buf87  # reuse
    buf394 = reinterpret_tensor(buf94, (128, ), (1, ), 0); del buf94  # reuse
    buf402 = reinterpret_tensor(buf104, (512, ), (1, ), 0); del buf104  # reuse
    buf410 = reinterpret_tensor(buf109, (512, ), (1, ), 0); del buf109  # reuse
    buf418 = reinterpret_tensor(buf116, (128, ), (1, ), 0); del buf116  # reuse
    buf426 = reinterpret_tensor(buf123, (128, ), (1, ), 0); del buf123  # reuse
    buf434 = reinterpret_tensor(buf133, (512, ), (1, ), 0); del buf133  # reuse
    buf442 = reinterpret_tensor(buf140, (256, ), (1, ), 0); del buf140  # reuse
    buf450 = reinterpret_tensor(buf147, (256, ), (1, ), 0); del buf147  # reuse
    buf458 = reinterpret_tensor(buf157, (1024, ), (1, ), 0); del buf157  # reuse
    buf466 = reinterpret_tensor(buf162, (1024, ), (1, ), 0); del buf162  # reuse
    buf474 = reinterpret_tensor(buf169, (256, ), (1, ), 0); del buf169  # reuse
    buf482 = reinterpret_tensor(buf189, (256, ), (1, ), 0); del buf189  # reuse
    buf490 = reinterpret_tensor(buf196, (1024, ), (1, ), 0); del buf196  # reuse
    buf498 = reinterpret_tensor(buf203, (512, ), (1, ), 0); del buf203  # reuse
    buf506 = reinterpret_tensor(buf225, (512, ), (1, ), 0); del buf225  # reuse
    buf514 = reinterpret_tensor(buf232, (2048, ), (1, ), 0); del buf232  # reuse
    buf522 = reinterpret_tensor(buf237, (2048, ), (1, ), 0); del buf237  # reuse
    buf530 = reinterpret_tensor(buf244, (512, ), (1, ), 0); del buf244  # reuse
    buf538 = reinterpret_tensor(buf264, (512, ), (1, ), 0); del buf264  # reuse
    buf546 = reinterpret_tensor(buf271, (2048, ), (1, ), 0); del buf271  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_49(c_void_p(buf278.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()))
    del buf306
    del buf314
    del buf322
    del buf330
    del buf338
    del buf346
    del buf354
    del buf362
    del buf370
    del buf378
    del buf386
    del buf394
    del buf402
    del buf410
    del buf418
    del buf426
    del buf434
    del buf442
    del buf450
    del buf458
    del buf466
    del buf474
    del buf482
    del buf490
    del buf498
    del buf506
    del buf514
    del buf522
    del buf530
    del buf538
    del buf546
    del primals_107
    del primals_108
    del primals_109
    del primals_110
    del primals_111
    del primals_112
    del primals_113
    del primals_114
    del primals_115
    del primals_116
    del primals_117
    del primals_118
    del primals_119
    del primals_120
    del primals_121
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
    return (buf277, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, buf0, buf1, buf2, primals_72, buf3, primals_74, primals_75, primals_76, primals_77, buf4, primals_79, primals_80, primals_81, buf5, primals_83, primals_84, primals_85, primals_86, buf6, primals_88, primals_89, primals_90, buf7, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, buf8, buf9, buf13, buf15, buf16, buf20, buf22, buf23, buf27, buf29, buf30, buf31, buf32, buf36, buf38, buf39, buf43, buf44, buf46, buf47, buf48, buf49, buf53, buf54, buf58, buf60, buf61, buf65, buf67, buf68, buf72, buf73, buf75, buf76, buf77, buf78, buf82, buf84, buf85, buf89, buf91, buf92, buf96, buf97, buf99, buf100, buf101, buf102, buf106, buf107, buf111, buf113, buf114, buf118, buf120, buf121, buf125, buf126, buf128, buf129, buf130, buf131, buf135, buf137, buf138, buf142, buf144, buf145, buf149, buf150, buf152, buf153, buf154, buf155, buf159, buf160, buf164, buf166, buf167, buf171, buf173, buf178, buf180, buf187, buf191, buf193, buf194, buf198, buf200, buf201, buf205, buf207, buf212, buf214, buf222, buf223, buf227, buf229, buf230, buf234, buf235, buf239, buf241, buf242, buf246, buf248, buf253, buf255, buf262, buf266, buf268, buf269, buf273, buf276, reinterpret_tensor(primals_105, (1000, 2048), (2048, 1), 0), buf278, reinterpret_tensor(buf270, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf279, reinterpret_tensor(buf263, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf260, (32, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf261, (32, 128, 64), (8192, 64, 1), 0), buf260, reinterpret_tensor(primals_64, (15, 16), (16, 1), 0), reinterpret_tensor(primals_63, (15, 16), (16, 1), 0), reinterpret_tensor(buf250, (32, 16, 64), (1024, 64, 1), 0), reinterpret_tensor(buf251, (32, 64, 16), (1024, 1, 64), 0), buf280, reinterpret_tensor(buf243, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf281, reinterpret_tensor(buf236, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf231, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf282, reinterpret_tensor(buf224, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf219, (32, 256, 256), (65536, 1, 256), 0), reinterpret_tensor(buf220, (32, 128, 256), (32768, 256, 1), 0), buf219, reinterpret_tensor(primals_54, (31, 16), (16, 1), 0), reinterpret_tensor(primals_53, (31, 16), (16, 1), 0), reinterpret_tensor(buf209, (32, 16, 256), (4096, 256, 1), 0), reinterpret_tensor(buf210, (32, 256, 16), (4096, 1, 256), 0), buf283, reinterpret_tensor(buf202, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf284, reinterpret_tensor(buf195, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf285, reinterpret_tensor(buf188, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf185, (32, 256, 256), (65536, 1, 256), 0), reinterpret_tensor(buf186, (32, 64, 256), (16384, 256, 1), 0), buf185, reinterpret_tensor(primals_46, (31, 16), (16, 1), 0), reinterpret_tensor(primals_45, (31, 16), (16, 1), 0), reinterpret_tensor(buf175, (32, 16, 256), (4096, 256, 1), 0), reinterpret_tensor(buf176, (32, 256, 16), (4096, 1, 256), 0), buf286, reinterpret_tensor(buf168, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf287, reinterpret_tensor(buf161, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf156, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf146, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf288, reinterpret_tensor(buf139, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf289, reinterpret_tensor(buf132, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf122, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf290, reinterpret_tensor(buf115, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf291, reinterpret_tensor(buf108, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf103, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf93, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf292, reinterpret_tensor(buf86, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf293, reinterpret_tensor(buf79, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf69, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf294, reinterpret_tensor(buf62, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf295, reinterpret_tensor(buf55, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf50, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf40, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf296, reinterpret_tensor(buf33, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf297, reinterpret_tensor(buf24, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf298, reinterpret_tensor(buf17, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf299, reinterpret_tensor(buf10, (1, 24, 1, 1), (24, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((31, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((31, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((31, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((31, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((15, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((15, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((1, 1, 3), (3, 3, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1, 1, 3), (3, 3, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1, 1, 5), (5, 5, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((1, 1, 5), (5, 5, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((256, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1, 1, 5), (5, 5, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((384, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_108 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_111 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_114 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_117 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_120 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_123 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_126 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_129 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_132 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_135 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_138 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_141 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_144 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_147 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_150 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_153 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_156 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_159 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_162 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_165 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_168 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_171 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_174 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_177 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_180 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_183 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_186 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_189 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_192 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_195 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_198 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('eca_botnext26ts_256', benchmark_compiled_module)
