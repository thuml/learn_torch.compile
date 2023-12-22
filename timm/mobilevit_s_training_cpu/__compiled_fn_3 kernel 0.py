
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (96L*x2) + (864L*x0)), static_cast<long>(96L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (96L*x2) + (864L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2880L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (320L*x2) + (2880L*x0)), static_cast<long>(320L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2880L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (320L*x2) + (2880L*x0)));
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
                        auto tmp0 = in_ptr7[static_cast<long>(x2 + (65536L*x1) + (196608L*x0))];
                        out_ptr7[static_cast<long>(x1 + (3L*x2) + (196608L*x0))] = tmp0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (16L*x0)));
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
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_3 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_4 = async_compile.cpp('''
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_5 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16777216L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_silu_6 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_functional_7 = async_compile.cpp('''
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
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_8 = async_compile.cpp('''
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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


cpp_fused__native_batch_norm_legit_functional_add_10 = async_compile.cpp('''
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
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_11 = async_compile.cpp('''
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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


cpp_fused__native_batch_norm_legit_functional_silu_12 = async_compile.cpp('''
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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


cpp_fused__native_batch_norm_legit_functional_add_13 = async_compile.cpp('''
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
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_14 = async_compile.cpp('''
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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


cpp_fused__native_batch_norm_legit_functional_silu_15 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_functional_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused_native_layer_norm_view_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (147456L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (147456L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (147456L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (147456L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (147456L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (147456L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(144.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (144L*x0) + (36864L*x1) + (36864L*x1_inner))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.m2);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(144.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        tmp13.store(out_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_view_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2359296L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.m2);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(144.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-05);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.m2);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(144.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2359296L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(144.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = out_ptr2[static_cast<long>((144L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (64L*(c10::div_floor_integer((x2 + (32L*x1)), 64L))) + (static_cast<long>(x2) % static_cast<long>(2L))), 4L)) % static_cast<long>(256L))) + (36864L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>(x2) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (64L*(c10::div_floor_integer((x2 + (32L*x1)), 64L))) + (1024L*x3) + (1024L*x3_inner) + (147456L*x0) + (static_cast<long>(x2) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (64L*(c10::div_floor_integer((x2 + (32L*x1)), 64L))) + (1024L*x3) + (1024L*x3_inner) + (static_cast<long>(x2) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr4[static_cast<long>(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (64L*(c10::div_floor_integer((x2 + (32L*x1)), 64L))) + (1024L*x3) + (1024L*x3_inner) + (static_cast<long>(x2) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr5[static_cast<long>(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (64L*(c10::div_floor_integer((x2 + (32L*x1)), 64L))) + (1024L*x3) + (1024L*x3_inner) + (static_cast<long>(x2) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr3 + static_cast<long>(x3 + (144L*x2) + (4608L*x1) + (147456L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(96);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + (96L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(192);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = out_ptr3[static_cast<long>((-96L) + x1 + (96L*x0))];
                        auto tmp13 = decltype(tmp12)(1) / (decltype(tmp12)(1) + std::exp(-tmp12));
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp16 = tmp4 ? tmp7 : tmp15;
                    out_ptr4[static_cast<long>(x1 + (192L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
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


cpp_fused__native_batch_norm_legit_functional_silu_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_29 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_30 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (49152L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (49152L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (49152L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (49152L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (49152L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (49152L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(192.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (192L*x0) + (12288L*x1) + (12288L*x1_inner))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.m2);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (64L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(192.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        tmp13.store(out_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.m2);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x1 + (64L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(192.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-05);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0.m2);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (64L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(192.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(192.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(192.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_view_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(192.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(192.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(192.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = out_ptr2[static_cast<long>((192L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (32L*(c10::div_floor_integer((x2 + (16L*x1)), 32L))) + (static_cast<long>(x2) % static_cast<long>(2L))), 4L)) % static_cast<long>(64L))) + (12288L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>(x2) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (32L*(c10::div_floor_integer((x2 + (16L*x1)), 32L))) + (256L*x3) + (256L*x3_inner) + (49152L*x0) + (static_cast<long>(x2) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (32L*(c10::div_floor_integer((x2 + (16L*x1)), 32L))) + (256L*x3) + (256L*x3_inner) + (static_cast<long>(x2) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr4[static_cast<long>(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (32L*(c10::div_floor_integer((x2 + (16L*x1)), 32L))) + (256L*x3) + (256L*x3_inner) + (static_cast<long>(x2) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr5[static_cast<long>(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (32L*(c10::div_floor_integer((x2 + (16L*x1)), 32L))) + (256L*x3) + (256L*x3_inner) + (static_cast<long>(x2) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr3 + static_cast<long>(x3 + (192L*x2) + (3072L*x1) + (49152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + (128L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(256);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = out_ptr3[static_cast<long>((-128L) + x1 + (128L*x0))];
                        auto tmp13 = decltype(tmp12)(1) / (decltype(tmp12)(1) + std::exp(-tmp12));
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp16 = tmp4 ? tmp7 : tmp15;
                    out_ptr4[static_cast<long>(x1 + (256L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_45 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_46 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_functional_silu_47 = async_compile.cpp('''
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
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_48 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_49 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(81920L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr4 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (15360L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (15360L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (15360L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (15360L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (15360L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (15360L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(240.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (240L*x0) + (3840L*x1) + (3840L*x1_inner))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (16L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (16L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(240.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        tmp13.store(out_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_view_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(245760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (16L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x1 + (16L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(240.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-05);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (16L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (16L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(240.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(245760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(240.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(240.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_silu_view_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(245760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(240.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = out_ptr2[static_cast<long>((240L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (16L*(c10::div_floor_integer((x2 + (8L*x1)), 16L))) + (static_cast<long>(x2) % static_cast<long>(2L))), 4L)) % static_cast<long>(16L))) + (3840L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>(x2) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (16L*(c10::div_floor_integer((x2 + (8L*x1)), 16L))) + (64L*x3) + (64L*x3_inner) + (15360L*x0) + (static_cast<long>(x2) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (16L*(c10::div_floor_integer((x2 + (8L*x1)), 16L))) + (64L*x3) + (64L*x3_inner) + (static_cast<long>(x2) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr3[static_cast<long>(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (16L*(c10::div_floor_integer((x2 + (8L*x1)), 16L))) + (64L*x3) + (64L*x3_inner) + (static_cast<long>(x2) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr4[static_cast<long>(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer(x2, 2L))) + (16L*(c10::div_floor_integer((x2 + (8L*x1)), 16L))) + (64L*x3) + (64L*x3_inner) + (static_cast<long>(x2) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr3 + static_cast<long>(x3 + (240L*x2) + (1920L*x1) + (15360L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_60 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(160);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + (160L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(320);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = out_ptr3[static_cast<long>((-160L) + x1 + (160L*x0))];
                        auto tmp13 = decltype(tmp12)(1) / (decltype(tmp12)(1) + std::exp(-tmp12));
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp16 = tmp4 ? tmp7 : tmp15;
                    out_ptr4[static_cast<long>(x1 + (320L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_61 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(81920L); x0+=static_cast<long>(8L))
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
''')


cpp_fused__native_batch_norm_legit_functional_mean_silu_view_62 = async_compile.cpp('''
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (640L*x2) + (40960L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5120L); x0+=static_cast<long>(8L))
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


cpp_fused__native_batch_norm_legit_functional_add_detach_fill_mul_native_layer_norm_native_layer_norm_backward_sigmoid_sub_63 = async_compile.cpp('''
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
                       const long* in_ptr124,
                       const float* in_ptr125,
                       const float* in_ptr126,
                       const float* in_ptr127,
                       const long* in_ptr128,
                       const float* in_ptr129,
                       const float* in_ptr130,
                       const float* in_ptr131,
                       const long* in_ptr132,
                       const float* in_ptr133,
                       const float* in_ptr134,
                       const float* in_ptr135,
                       const long* in_ptr136,
                       const float* in_ptr137,
                       const float* in_ptr138,
                       const float* in_ptr139,
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
                       long* out_ptr13,
                       float* out_ptr15,
                       float* out_ptr16,
                       long* out_ptr18,
                       float* out_ptr20,
                       float* out_ptr21,
                       long* out_ptr23,
                       float* out_ptr25,
                       float* out_ptr26,
                       long* out_ptr28,
                       float* out_ptr30,
                       float* out_ptr31,
                       long* out_ptr33,
                       float* out_ptr35,
                       float* out_ptr36,
                       long* out_ptr38,
                       float* out_ptr40,
                       float* out_ptr41,
                       long* out_ptr43,
                       float* out_ptr45,
                       float* out_ptr46,
                       long* out_ptr48,
                       float* out_ptr50,
                       float* out_ptr51,
                       long* out_ptr53,
                       float* out_ptr55,
                       float* out_ptr56,
                       long* out_ptr58,
                       float* out_ptr60,
                       float* out_ptr61,
                       long* out_ptr63,
                       float* out_ptr65,
                       float* out_ptr66,
                       long* out_ptr68,
                       float* out_ptr70,
                       float* out_ptr71,
                       long* out_ptr73,
                       float* out_ptr75,
                       float* out_ptr76,
                       long* out_ptr78,
                       float* out_ptr80,
                       float* out_ptr81,
                       long* out_ptr83,
                       float* out_ptr85,
                       float* out_ptr86,
                       long* out_ptr88,
                       float* out_ptr90,
                       float* out_ptr91,
                       long* out_ptr93,
                       float* out_ptr95,
                       float* out_ptr96,
                       long* out_ptr98,
                       float* out_ptr100,
                       float* out_ptr101,
                       long* out_ptr103,
                       float* out_ptr105,
                       float* out_ptr106,
                       long* out_ptr108,
                       float* out_ptr110,
                       float* out_ptr111,
                       long* out_ptr113,
                       float* out_ptr115,
                       float* out_ptr116,
                       long* out_ptr118,
                       float* out_ptr120,
                       float* out_ptr121,
                       long* out_ptr123,
                       float* out_ptr125,
                       float* out_ptr126,
                       long* out_ptr128,
                       float* out_ptr130,
                       float* out_ptr131,
                       long* out_ptr133,
                       float* out_ptr135,
                       float* out_ptr136,
                       long* out_ptr138,
                       float* out_ptr140,
                       float* out_ptr141,
                       long* out_ptr143,
                       float* out_ptr145,
                       float* out_ptr146,
                       long* out_ptr148,
                       float* out_ptr150,
                       float* out_ptr151,
                       long* out_ptr153,
                       float* out_ptr155,
                       float* out_ptr156,
                       long* out_ptr158,
                       float* out_ptr160,
                       float* out_ptr161,
                       long* out_ptr163,
                       float* out_ptr165,
                       float* out_ptr166,
                       long* out_ptr168,
                       float* out_ptr170,
                       float* out_ptr171)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(81920L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(240.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(240.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(60L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (60L*x1) + (240L*x0))];
                        out_ptr0[static_cast<long>(x1 + (4L*x2) + (240L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(240.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(240.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(60L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (60L*x1) + (240L*x0))];
                        out_ptr1[static_cast<long>(x1 + (4L*x2) + (240L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(240.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(240.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(60L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr2[static_cast<long>(x2 + (60L*x1) + (240L*x0))];
                        out_ptr2[static_cast<long>(x1 + (4L*x2) + (240L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x0 + (32L*x1)), static_cast<long>(32L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            auto tmp2 = static_cast<float>(240.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(1e-05);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 + tmp6;
                            auto tmp8 = tmp7.rsqrt();
                            auto tmp9 = tmp8 / tmp3;
                            tmp9.store(out_ptr3 + static_cast<long>(x1 + (16L*x0) + (16L*x0_inner)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(81920L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x2 + (48L*x1) + (192L*x0))];
                        out_ptr4[static_cast<long>(x1 + (4L*x2) + (192L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(192.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr16 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x2 + (48L*x1) + (192L*x0))];
                        out_ptr5[static_cast<long>(x1 + (4L*x2) + (192L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr17 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(192.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr6[static_cast<long>(x2 + (48L*x1) + (192L*x0))];
                        out_ptr6[static_cast<long>(x1 + (4L*x2) + (192L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr7[static_cast<long>(x2 + (48L*x1) + (192L*x0))];
                        out_ptr7[static_cast<long>(x1 + (4L*x2) + (192L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr8 + static_cast<long>(x0 + (32L*x1)), static_cast<long>(32L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            auto tmp2 = static_cast<float>(192.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(1e-05);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 + tmp6;
                            auto tmp8 = tmp7.rsqrt();
                            auto tmp9 = tmp8 / tmp3;
                            tmp9.store(out_ptr8 + static_cast<long>(x1 + (64L*x0) + (64L*x0_inner)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr21 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr22 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr23 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr24 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr25 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr26 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr9[static_cast<long>(x2 + (36L*x1) + (144L*x0))];
                        out_ptr9[static_cast<long>(x1 + (4L*x2) + (144L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(144.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr10[static_cast<long>(x2 + (36L*x1) + (144L*x0))];
                        out_ptr10[static_cast<long>(x1 + (4L*x2) + (144L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr11 + static_cast<long>(x0 + (32L*x1)), static_cast<long>(32L), tmp0, 8);
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                            auto tmp2 = static_cast<float>(144.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(1e-05);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 + tmp6;
                            auto tmp8 = tmp7.rsqrt();
                            auto tmp9 = tmp8 / tmp3;
                            tmp9.store(out_ptr11 + static_cast<long>(x1 + (256L*x0) + (256L*x0_inner)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr30 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr31 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr32 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr33 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr34 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr35 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr36 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr37 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16777216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr38 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr39 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr40 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr41 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr12[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr13[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr16 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr16 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr16[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr18[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr20 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr21 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr21 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr20[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr23[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr25 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr26 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr26 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr24[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr28[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr30 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr31 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr31 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr28[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr33[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr35 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr35 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr36 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr36 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr32[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr38[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr33 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr40 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr40 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr41 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr41 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr36[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr43[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr37 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr45 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr45 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr46 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr46 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr40[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr48[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr41 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr50 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr50 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr51 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr51 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr44[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr53[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr45 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr55 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr55 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr56 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr56 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr48[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr58[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr49 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr60 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr60 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr61 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr61 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr52[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr63[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr53 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr65 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr65 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr66 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr66 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr56[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr68[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr57 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr70 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr70 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr71 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr71 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr60[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr73[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr61 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr75 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr75 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr76 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr76 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr64[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr78[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr65 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr80 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr80 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr55 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr81 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr81 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr68[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr83[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr69 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr85 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr85 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr86 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr86 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr72[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr88[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr73 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr90 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr90 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr57 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr91 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr91 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr76[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr93[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr77 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr95 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr95 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr58 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr96 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr96 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr80[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr98[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr81 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr100 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr100 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr59 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr101 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr101 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr84[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr103[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr85 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr105 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr105 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr60 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr106 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr106 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr88[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr108[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr89 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr110 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr110 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr61 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr111 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr111 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr92[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr113[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr93 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr115 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr115 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr62 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr116 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr116 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr96[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr118[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr97 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr120 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr120 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr63 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr121 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr121 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr100[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr123[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr101 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr125 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr125 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr64 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr126 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr126 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr104[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr128[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr105 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr130 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr130 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr65 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr131 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr131 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr108[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr133[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr109 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr135 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr135 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr66 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr136 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr136 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr112[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr138[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr113 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr140 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr140 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr67 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr141 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr141 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr116[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr143[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr117 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr145 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr145 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr68 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr146 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr146 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr120[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr148[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr121 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr150 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr150 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr69 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr151 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr151 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr124[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr153[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr125 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr155 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr155 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr70 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr156 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr156 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr128[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr158[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr129 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr160 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr160 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr71 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr161 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr161 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr132[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr163[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr133 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr165 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr165 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr72 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr166 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr166 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr136[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr168[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr137 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr170 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr170 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr73 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr171 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr171 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (96, ), (1, ))
    assert_size_stride(primals_32, (96, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_34, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_36, (96, ), (1, ))
    assert_size_stride(primals_37, (96, ), (1, ))
    assert_size_stride(primals_38, (96, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_42, (384, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (160, ), (1, ))
    assert_size_stride(primals_56, (160, ), (1, ))
    assert_size_stride(primals_57, (160, ), (1, ))
    assert_size_stride(primals_58, (160, ), (1, ))
    assert_size_stride(primals_59, (160, ), (1, ))
    assert_size_stride(primals_60, (160, ), (1, ))
    assert_size_stride(primals_61, (160, ), (1, ))
    assert_size_stride(primals_62, (160, ), (1, ))
    assert_size_stride(primals_63, (640, ), (1, ))
    assert_size_stride(primals_64, (640, ), (1, ))
    assert_size_stride(primals_65, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_66, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_67, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_69, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_70, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_71, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_72, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_75, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_76, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_77, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_78, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_79, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_80, (96, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_81, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_82, (144, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_83, (144, ), (1, ))
    assert_size_stride(primals_84, (144, ), (1, ))
    assert_size_stride(primals_85, (432, 144), (144, 1))
    assert_size_stride(primals_86, (432, ), (1, ))
    assert_size_stride(primals_87, (144, 144), (144, 1))
    assert_size_stride(primals_88, (144, ), (1, ))
    assert_size_stride(primals_89, (144, ), (1, ))
    assert_size_stride(primals_90, (144, ), (1, ))
    assert_size_stride(primals_91, (288, 144), (144, 1))
    assert_size_stride(primals_92, (288, ), (1, ))
    assert_size_stride(primals_93, (144, 288), (288, 1))
    assert_size_stride(primals_94, (144, ), (1, ))
    assert_size_stride(primals_95, (144, ), (1, ))
    assert_size_stride(primals_96, (144, ), (1, ))
    assert_size_stride(primals_97, (432, 144), (144, 1))
    assert_size_stride(primals_98, (432, ), (1, ))
    assert_size_stride(primals_99, (144, 144), (144, 1))
    assert_size_stride(primals_100, (144, ), (1, ))
    assert_size_stride(primals_101, (144, ), (1, ))
    assert_size_stride(primals_102, (144, ), (1, ))
    assert_size_stride(primals_103, (288, 144), (144, 1))
    assert_size_stride(primals_104, (288, ), (1, ))
    assert_size_stride(primals_105, (144, 288), (288, 1))
    assert_size_stride(primals_106, (144, ), (1, ))
    assert_size_stride(primals_107, (144, ), (1, ))
    assert_size_stride(primals_108, (144, ), (1, ))
    assert_size_stride(primals_109, (96, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_110, (96, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_111, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_112, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_114, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_115, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_116, (192, ), (1, ))
    assert_size_stride(primals_117, (192, ), (1, ))
    assert_size_stride(primals_118, (576, 192), (192, 1))
    assert_size_stride(primals_119, (576, ), (1, ))
    assert_size_stride(primals_120, (192, 192), (192, 1))
    assert_size_stride(primals_121, (192, ), (1, ))
    assert_size_stride(primals_122, (192, ), (1, ))
    assert_size_stride(primals_123, (192, ), (1, ))
    assert_size_stride(primals_124, (384, 192), (192, 1))
    assert_size_stride(primals_125, (384, ), (1, ))
    assert_size_stride(primals_126, (192, 384), (384, 1))
    assert_size_stride(primals_127, (192, ), (1, ))
    assert_size_stride(primals_128, (192, ), (1, ))
    assert_size_stride(primals_129, (192, ), (1, ))
    assert_size_stride(primals_130, (576, 192), (192, 1))
    assert_size_stride(primals_131, (576, ), (1, ))
    assert_size_stride(primals_132, (192, 192), (192, 1))
    assert_size_stride(primals_133, (192, ), (1, ))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_135, (192, ), (1, ))
    assert_size_stride(primals_136, (384, 192), (192, 1))
    assert_size_stride(primals_137, (384, ), (1, ))
    assert_size_stride(primals_138, (192, 384), (384, 1))
    assert_size_stride(primals_139, (192, ), (1, ))
    assert_size_stride(primals_140, (192, ), (1, ))
    assert_size_stride(primals_141, (192, ), (1, ))
    assert_size_stride(primals_142, (576, 192), (192, 1))
    assert_size_stride(primals_143, (576, ), (1, ))
    assert_size_stride(primals_144, (192, 192), (192, 1))
    assert_size_stride(primals_145, (192, ), (1, ))
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_147, (192, ), (1, ))
    assert_size_stride(primals_148, (384, 192), (192, 1))
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (192, 384), (384, 1))
    assert_size_stride(primals_151, (192, ), (1, ))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_153, (192, ), (1, ))
    assert_size_stride(primals_154, (576, 192), (192, 1))
    assert_size_stride(primals_155, (576, ), (1, ))
    assert_size_stride(primals_156, (192, 192), (192, 1))
    assert_size_stride(primals_157, (192, ), (1, ))
    assert_size_stride(primals_158, (192, ), (1, ))
    assert_size_stride(primals_159, (192, ), (1, ))
    assert_size_stride(primals_160, (384, 192), (192, 1))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (192, 384), (384, 1))
    assert_size_stride(primals_163, (192, ), (1, ))
    assert_size_stride(primals_164, (192, ), (1, ))
    assert_size_stride(primals_165, (192, ), (1, ))
    assert_size_stride(primals_166, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_167, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_168, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_169, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_170, (160, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_171, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_172, (240, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_173, (240, ), (1, ))
    assert_size_stride(primals_174, (240, ), (1, ))
    assert_size_stride(primals_175, (720, 240), (240, 1))
    assert_size_stride(primals_176, (720, ), (1, ))
    assert_size_stride(primals_177, (240, 240), (240, 1))
    assert_size_stride(primals_178, (240, ), (1, ))
    assert_size_stride(primals_179, (240, ), (1, ))
    assert_size_stride(primals_180, (240, ), (1, ))
    assert_size_stride(primals_181, (480, 240), (240, 1))
    assert_size_stride(primals_182, (480, ), (1, ))
    assert_size_stride(primals_183, (240, 480), (480, 1))
    assert_size_stride(primals_184, (240, ), (1, ))
    assert_size_stride(primals_185, (240, ), (1, ))
    assert_size_stride(primals_186, (240, ), (1, ))
    assert_size_stride(primals_187, (720, 240), (240, 1))
    assert_size_stride(primals_188, (720, ), (1, ))
    assert_size_stride(primals_189, (240, 240), (240, 1))
    assert_size_stride(primals_190, (240, ), (1, ))
    assert_size_stride(primals_191, (240, ), (1, ))
    assert_size_stride(primals_192, (240, ), (1, ))
    assert_size_stride(primals_193, (480, 240), (240, 1))
    assert_size_stride(primals_194, (480, ), (1, ))
    assert_size_stride(primals_195, (240, 480), (480, 1))
    assert_size_stride(primals_196, (240, ), (1, ))
    assert_size_stride(primals_197, (240, ), (1, ))
    assert_size_stride(primals_198, (240, ), (1, ))
    assert_size_stride(primals_199, (720, 240), (240, 1))
    assert_size_stride(primals_200, (720, ), (1, ))
    assert_size_stride(primals_201, (240, 240), (240, 1))
    assert_size_stride(primals_202, (240, ), (1, ))
    assert_size_stride(primals_203, (240, ), (1, ))
    assert_size_stride(primals_204, (240, ), (1, ))
    assert_size_stride(primals_205, (480, 240), (240, 1))
    assert_size_stride(primals_206, (480, ), (1, ))
    assert_size_stride(primals_207, (240, 480), (480, 1))
    assert_size_stride(primals_208, (240, ), (1, ))
    assert_size_stride(primals_209, (240, ), (1, ))
    assert_size_stride(primals_210, (240, ), (1, ))
    assert_size_stride(primals_211, (160, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_212, (160, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_213, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_214, (1000, 640), (640, 1))
    assert_size_stride(primals_215, (1000, ), (1, ))
    assert_size_stride(primals_216, (), ())
    assert_size_stride(primals_217, (16, ), (1, ))
    assert_size_stride(primals_218, (16, ), (1, ))
    assert_size_stride(primals_219, (), ())
    assert_size_stride(primals_220, (64, ), (1, ))
    assert_size_stride(primals_221, (64, ), (1, ))
    assert_size_stride(primals_222, (), ())
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (64, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (32, ), (1, ))
    assert_size_stride(primals_227, (32, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (128, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (128, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (64, ), (1, ))
    assert_size_stride(primals_236, (64, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (256, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (256, ), (1, ))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (64, ), (1, ))
    assert_size_stride(primals_245, (64, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (256, ), (1, ))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (64, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (256, ), (1, ))
    assert_size_stride(primals_257, (256, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (256, ), (1, ))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (96, ), (1, ))
    assert_size_stride(primals_263, (96, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (96, ), (1, ))
    assert_size_stride(primals_266, (96, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (96, ), (1, ))
    assert_size_stride(primals_269, (96, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (96, ), (1, ))
    assert_size_stride(primals_272, (96, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (384, ), (1, ))
    assert_size_stride(primals_275, (384, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (384, ), (1, ))
    assert_size_stride(primals_278, (384, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (128, ), (1, ))
    assert_size_stride(primals_284, (128, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (128, ), (1, ))
    assert_size_stride(primals_290, (128, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (512, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (160, ), (1, ))
    assert_size_stride(primals_299, (160, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (160, ), (1, ))
    assert_size_stride(primals_302, (160, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (160, ), (1, ))
    assert_size_stride(primals_305, (160, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (160, ), (1, ))
    assert_size_stride(primals_308, (160, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (640, ), (1, ))
    assert_size_stride(primals_311, (640, ), (1, ))
    assert_size_stride(primals_312, (8, 3, 256, 256), (196608, 65536, 256, 1))
    buf0 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((96, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((160, 320, 3, 3), (2880, 1, 960, 320), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_65.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del primals_110
    del primals_114
    del primals_167
    del primals_171
    del primals_212
    del primals_312
    del primals_65
    del primals_81
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf7, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 16, 128, 128), (262144, 1, 2048, 16))
    buf9 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf12 = empty((16, ), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 16, 128, 128), (262144, 1, 2048, 16), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((8, 16, 128, 128), (262144, 1, 2048, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_1(c_void_p(buf8.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()))
    del primals_2
    # Source Nodes: [x_6], Original ATen: [aten.convolution]
    buf15 = extern_kernels.convolution(buf14, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf15, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    buf16 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf19 = empty((64, ), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_2(c_void_p(buf15.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    del primals_4
    # Source Nodes: [x_12], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(buf21, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf22, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    buf23 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf26 = empty((64, ), device='cpu', dtype=torch.float32)
    buf27 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_3(c_void_p(buf22.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    del primals_6
    # Source Nodes: [x_20], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(buf28, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf29, (8, 32, 128, 128), (524288, 1, 4096, 32))
    buf30 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf33 = empty((32, ), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_4(c_void_p(buf29.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del primals_8
    # Source Nodes: [x_28], Original ATen: [aten.convolution]
    buf35 = extern_kernels.convolution(buf34, primals_69, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf35, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    buf36 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf37 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf39 = empty((128, ), device='cpu', dtype=torch.float32)
    buf40 = empty_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_5(c_void_p(buf35.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    del primals_10
    # Source Nodes: [x_34], Original ATen: [aten.convolution]
    buf42 = extern_kernels.convolution(buf41, primals_70, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
    assert_size_stride(buf42, (8, 128, 64, 64), (524288, 1, 8192, 128))
    buf43 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf44 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf46 = empty((128, ), device='cpu', dtype=torch.float32)
    buf47 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    buf48 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_6(c_void_p(buf42.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del primals_12
    # Source Nodes: [x_42], Original ATen: [aten.convolution]
    buf49 = extern_kernels.convolution(buf48, primals_71, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 64, 64, 64), (262144, 1, 4096, 64))
    buf50 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf51 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf53 = empty((64, ), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_7(c_void_p(buf49.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del primals_14
    # Source Nodes: [x_50], Original ATen: [aten.convolution]
    buf55 = extern_kernels.convolution(buf54, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf55, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    buf56 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf57 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf59 = empty((256, ), device='cpu', dtype=torch.float32)
    buf60 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf61 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_8(c_void_p(buf55.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    del primals_16
    # Source Nodes: [x_56], Original ATen: [aten.convolution]
    buf62 = extern_kernels.convolution(buf61, primals_73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
    assert_size_stride(buf62, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    buf63 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf64 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf66 = empty((256, ), device='cpu', dtype=torch.float32)
    buf67 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf68 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_9(c_void_p(buf62.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del primals_18
    # Source Nodes: [x_64], Original ATen: [aten.convolution]
    buf69 = extern_kernels.convolution(buf68, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf69, (8, 64, 64, 64), (262144, 1, 4096, 64))
    buf70 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf71 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf73 = empty((64, ), device='cpu', dtype=torch.float32)
    buf74 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_10(c_void_p(buf69.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    del primals_20
    # Source Nodes: [x_73], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(buf74, primals_75, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf75, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    buf76 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf77 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf79 = empty((256, ), device='cpu', dtype=torch.float32)
    buf80 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf81 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_11(c_void_p(buf75.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del primals_22
    # Source Nodes: [x_79], Original ATen: [aten.convolution]
    buf82 = extern_kernels.convolution(buf81, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
    assert_size_stride(buf82, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    buf83 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf84 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf86 = empty((256, ), device='cpu', dtype=torch.float32)
    buf87 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf88 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_12(c_void_p(buf82.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    del primals_24
    # Source Nodes: [x_87], Original ATen: [aten.convolution]
    buf89 = extern_kernels.convolution(buf88, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf89, (8, 64, 64, 64), (262144, 1, 4096, 64))
    buf90 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf91 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf93 = empty((64, ), device='cpu', dtype=torch.float32)
    buf94 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_13(c_void_p(buf89.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del primals_26
    # Source Nodes: [x_96], Original ATen: [aten.convolution]
    buf95 = extern_kernels.convolution(buf94, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf95, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    buf96 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf99 = empty((256, ), device='cpu', dtype=torch.float32)
    buf100 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf101 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_14(c_void_p(buf95.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del primals_28
    # Source Nodes: [x_102], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, primals_79, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
    assert_size_stride(buf102, (8, 256, 32, 32), (262144, 1, 8192, 256))
    buf103 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf104 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf106 = empty((256, ), device='cpu', dtype=torch.float32)
    buf107 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    buf108 = empty_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_15(c_void_p(buf102.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()))
    del primals_30
    # Source Nodes: [x_110], Original ATen: [aten.convolution]
    buf109 = extern_kernels.convolution(buf108, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf109, (8, 96, 32, 32), (98304, 1, 3072, 96))
    buf110 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf111 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf113 = empty((96, ), device='cpu', dtype=torch.float32)
    buf114 = empty_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_16(c_void_p(buf109.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()))
    del primals_32
    # Source Nodes: [x_118], Original ATen: [aten.convolution]
    buf115 = extern_kernels.convolution(buf114, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf115, (8, 96, 32, 32), (98304, 1, 3072, 96))
    buf116 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf117 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf119 = empty((96, ), device='cpu', dtype=torch.float32)
    buf120 = empty_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    buf121 = empty_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_17(c_void_p(buf115.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    del primals_34
    # Source Nodes: [x_124], Original ATen: [aten.convolution]
    buf122 = extern_kernels.convolution(buf121, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf122, (8, 144, 32, 32), (147456, 1, 4608, 144))
    buf123 = empty_strided((32, 256, 1), (1, 32, 8192), device='cpu', dtype=torch.float32)
    buf124 = empty_strided((32, 256, 1), (1, 32, 8192), device='cpu', dtype=torch.float32)
    buf126 = empty((32, 256, 144), device='cpu', dtype=torch.float32)
    buf127 = empty((8192, 144), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_18(c_void_p(buf122.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    del primals_84
    buf128 = empty((8192, 432), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, buf127, reinterpret_tensor(primals_85, (144, 432), (1, 144), 0), alpha=1, beta=1, out=buf128)
    del primals_86
    # Source Nodes: [x_127], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf129 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf128, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf128, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf128, (32, 4, 256, 36), (110592, 36, 432, 1), 288))
    buf130 = buf129[0]
    buf131 = buf129[1]
    buf132 = buf129[2]
    buf133 = buf129[3]
    buf134 = buf129[6]
    buf135 = buf129[7]
    del buf129
    buf137 = empty((8192, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_88, reinterpret_tensor(buf130, (8192, 144), (144, 1), 0), reinterpret_tensor(primals_87, (144, 144), (1, 144), 0), alpha=1, beta=1, out=buf137)
    del primals_88
    buf138 = reinterpret_tensor(buf123, (32, 256, 1), (256, 1, 8192), 0); del buf123  # reuse
    buf139 = empty_strided((32, 256, 1), (256, 1, 8192), device='cpu', dtype=torch.float32)
    buf141 = empty((32, 256, 144), device='cpu', dtype=torch.float32)
    buf142 = empty((8192, 144), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_19(c_void_p(buf122.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    del primals_90
    buf143 = empty((8192, 288), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf142, reinterpret_tensor(primals_91, (144, 288), (1, 144), 0), alpha=1, beta=1, out=buf143)
    del primals_92
    buf144 = empty((8192, 288), device='cpu', dtype=torch.float32)
    cpp_fused_silu_view_20(c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    buf145 = empty((8192, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_136], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_94, buf144, reinterpret_tensor(primals_93, (288, 144), (1, 288), 0), alpha=1, beta=1, out=buf145)
    del primals_94
    buf146 = buf138; del buf138  # reuse
    buf147 = empty_strided((32, 256, 1), (256, 1, 8192), device='cpu', dtype=torch.float32)
    buf149 = empty((32, 256, 144), device='cpu', dtype=torch.float32)
    buf150 = empty((8192, 144), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_21(c_void_p(buf122.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    del primals_96
    buf151 = empty((8192, 432), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_98, buf150, reinterpret_tensor(primals_97, (144, 432), (1, 144), 0), alpha=1, beta=1, out=buf151)
    del primals_98
    # Source Nodes: [x_139], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf152 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf151, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf151, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf151, (32, 4, 256, 36), (110592, 36, 432, 1), 288))
    buf153 = buf152[0]
    buf154 = buf152[1]
    buf155 = buf152[2]
    buf156 = buf152[3]
    buf157 = buf152[6]
    buf158 = buf152[7]
    del buf152
    buf160 = empty((8192, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_100, reinterpret_tensor(buf153, (8192, 144), (144, 1), 0), reinterpret_tensor(primals_99, (144, 144), (1, 144), 0), alpha=1, beta=1, out=buf160)
    del primals_100
    buf161 = buf146; del buf146  # reuse
    buf162 = empty_strided((32, 256, 1), (256, 1, 8192), device='cpu', dtype=torch.float32)
    buf164 = empty((32, 256, 144), device='cpu', dtype=torch.float32)
    buf165 = empty((8192, 144), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_22(c_void_p(buf122.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del primals_102
    buf166 = empty((8192, 288), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf165, reinterpret_tensor(primals_103, (144, 288), (1, 144), 0), alpha=1, beta=1, out=buf166)
    del primals_104
    buf167 = empty((8192, 288), device='cpu', dtype=torch.float32)
    cpp_fused_silu_view_23(c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    buf168 = empty((8192, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_106, buf167, reinterpret_tensor(primals_105, (288, 144), (1, 288), 0), alpha=1, beta=1, out=buf168)
    del primals_106
    buf169 = reinterpret_tensor(buf168, (32, 256, 144), (36864, 144, 1), 0); del buf168  # reuse
    buf170 = buf161; del buf161  # reuse
    buf171 = empty_strided((32, 256, 1), (256, 1, 8192), device='cpu', dtype=torch.float32)
    buf173 = empty((32, 256, 144), device='cpu', dtype=torch.float32)
    buf174 = empty_strided((8, 144, 32, 32), (147456, 1, 4608, 144), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_24(c_void_p(buf169.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del primals_108
    # Source Nodes: [x_156], Original ATen: [aten.convolution]
    buf175 = extern_kernels.convolution(buf174, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf175, (8, 96, 32, 32), (98304, 1, 3072, 96))
    buf176 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf177 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf179 = empty((96, ), device='cpu', dtype=torch.float32)
    buf180 = empty_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    buf181 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_cat_25(c_void_p(buf175.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del primals_36
    # Source Nodes: [x_162], Original ATen: [aten.convolution]
    buf182 = extern_kernels.convolution(buf181, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf182, (8, 96, 32, 32), (98304, 1, 3072, 96))
    buf183 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf184 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf186 = empty((96, ), device='cpu', dtype=torch.float32)
    buf187 = empty_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    buf188 = empty_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_26(c_void_p(buf182.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    del primals_38
    # Source Nodes: [x_168], Original ATen: [aten.convolution]
    buf189 = extern_kernels.convolution(buf188, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf189, (8, 384, 32, 32), (393216, 1, 12288, 384))
    buf190 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf191 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf193 = empty((384, ), device='cpu', dtype=torch.float32)
    buf194 = empty_strided((8, 384, 32, 32), (393216, 1, 12288, 384), device='cpu', dtype=torch.float32)
    buf195 = empty_strided((8, 384, 32, 32), (393216, 1, 12288, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_27(c_void_p(buf189.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    del primals_40
    # Source Nodes: [x_174], Original ATen: [aten.convolution]
    buf196 = extern_kernels.convolution(buf195, primals_112, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
    assert_size_stride(buf196, (8, 384, 16, 16), (98304, 1, 6144, 384))
    buf197 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf198 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf200 = empty((384, ), device='cpu', dtype=torch.float32)
    buf201 = empty_strided((8, 384, 16, 16), (98304, 1, 6144, 384), device='cpu', dtype=torch.float32)
    buf202 = empty_strided((8, 384, 16, 16), (98304, 1, 6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_28(c_void_p(buf196.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    del primals_42
    # Source Nodes: [x_182], Original ATen: [aten.convolution]
    buf203 = extern_kernels.convolution(buf202, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf203, (8, 128, 16, 16), (32768, 1, 2048, 128))
    buf204 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf205 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf207 = empty((128, ), device='cpu', dtype=torch.float32)
    buf208 = empty_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_29(c_void_p(buf203.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    del primals_44
    # Source Nodes: [x_190], Original ATen: [aten.convolution]
    buf209 = extern_kernels.convolution(buf208, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf209, (8, 128, 16, 16), (32768, 1, 2048, 128))
    buf210 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf213 = empty((128, ), device='cpu', dtype=torch.float32)
    buf214 = empty_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    buf215 = empty_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_30(c_void_p(buf209.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    del primals_46
    # Source Nodes: [x_196], Original ATen: [aten.convolution]
    buf216 = extern_kernels.convolution(buf215, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf216, (8, 192, 16, 16), (49152, 1, 3072, 192))
    buf217 = empty_strided((32, 64, 1), (1, 32, 2048), device='cpu', dtype=torch.float32)
    buf218 = empty_strided((32, 64, 1), (1, 32, 2048), device='cpu', dtype=torch.float32)
    buf220 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    buf221 = empty((2048, 192), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_31(c_void_p(buf216.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    del primals_117
    buf222 = reinterpret_tensor(buf169, (2048, 576), (576, 1), 0); del buf169  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_119, buf221, reinterpret_tensor(primals_118, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf222)
    del primals_119
    # Source Nodes: [x_199], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf223 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf222, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf222, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf222, (32, 4, 64, 48), (36864, 48, 576, 1), 384))
    buf224 = buf223[0]
    buf225 = buf223[1]
    buf226 = buf223[2]
    buf227 = buf223[3]
    buf228 = buf223[6]
    buf229 = buf223[7]
    del buf223
    buf231 = empty((2048, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_201], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_121, reinterpret_tensor(buf224, (2048, 192), (192, 1), 0), reinterpret_tensor(primals_120, (192, 192), (1, 192), 0), alpha=1, beta=1, out=buf231)
    del primals_121
    buf232 = reinterpret_tensor(buf217, (32, 64, 1), (64, 1, 2048), 0); del buf217  # reuse
    buf233 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf235 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    buf236 = empty((2048, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_32(c_void_p(buf216.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    del primals_123
    buf237 = empty((2048, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_125, buf236, reinterpret_tensor(primals_124, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf237)
    del primals_125
    buf238 = empty((2048, 384), device='cpu', dtype=torch.float32)
    cpp_fused_silu_view_33(c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    buf239 = empty((2048, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_208], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_127, buf238, reinterpret_tensor(primals_126, (384, 192), (1, 384), 0), alpha=1, beta=1, out=buf239)
    del primals_127
    buf240 = buf232; del buf232  # reuse
    buf241 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf243 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    buf244 = empty((2048, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_34(c_void_p(buf216.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()))
    del primals_129
    buf245 = reinterpret_tensor(buf160, (2048, 576), (576, 1), 0); del buf160  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_131, buf244, reinterpret_tensor(primals_130, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf245)
    del primals_131
    # Source Nodes: [x_211], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf246 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf245, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf245, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf245, (32, 4, 64, 48), (36864, 48, 576, 1), 384))
    buf247 = buf246[0]
    buf248 = buf246[1]
    buf249 = buf246[2]
    buf250 = buf246[3]
    buf251 = buf246[6]
    buf252 = buf246[7]
    del buf246
    buf254 = empty((2048, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_213], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_133, reinterpret_tensor(buf247, (2048, 192), (192, 1), 0), reinterpret_tensor(primals_132, (192, 192), (1, 192), 0), alpha=1, beta=1, out=buf254)
    del primals_133
    buf255 = buf240; del buf240  # reuse
    buf256 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf258 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    buf259 = empty((2048, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_35(c_void_p(buf216.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    del primals_135
    buf260 = empty((2048, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_216], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_137, buf259, reinterpret_tensor(primals_136, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf260)
    del primals_137
    buf261 = empty((2048, 384), device='cpu', dtype=torch.float32)
    cpp_fused_silu_view_36(c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    buf262 = empty((2048, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_220], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_139, buf261, reinterpret_tensor(primals_138, (384, 192), (1, 384), 0), alpha=1, beta=1, out=buf262)
    del primals_139
    buf263 = reinterpret_tensor(buf262, (32, 64, 192), (12288, 192, 1), 0); del buf262  # reuse
    buf264 = buf255; del buf255  # reuse
    buf265 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf267 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    buf268 = empty((2048, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_37(c_void_p(buf263.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    del primals_141
    buf269 = reinterpret_tensor(buf145, (2048, 576), (576, 1), 0); del buf145  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_143, buf268, reinterpret_tensor(primals_142, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf269)
    del primals_143
    # Source Nodes: [x_223], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf270 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf269, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf269, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf269, (32, 4, 64, 48), (36864, 48, 576, 1), 384))
    buf271 = buf270[0]
    buf272 = buf270[1]
    buf273 = buf270[2]
    buf274 = buf270[3]
    buf275 = buf270[6]
    buf276 = buf270[7]
    del buf270
    buf278 = buf254; del buf254  # reuse
    # Source Nodes: [x_225], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_145, reinterpret_tensor(buf271, (2048, 192), (192, 1), 0), reinterpret_tensor(primals_144, (192, 192), (1, 192), 0), alpha=1, beta=1, out=buf278)
    del primals_145
    buf279 = buf264; del buf264  # reuse
    buf280 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf282 = reinterpret_tensor(buf239, (32, 64, 192), (12288, 192, 1), 0); del buf239  # reuse
    buf283 = buf231; del buf231  # reuse
    cpp_fused_add_native_layer_norm_view_38(c_void_p(buf263.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    del primals_147
    buf284 = empty((2048, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_228], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_149, buf283, reinterpret_tensor(primals_148, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf284)
    del primals_149
    buf285 = empty((2048, 384), device='cpu', dtype=torch.float32)
    cpp_fused_silu_view_39(c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    buf286 = reinterpret_tensor(buf216, (2048, 192), (192, 1), 0); del buf216  # reuse
    # Source Nodes: [x_232], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_151, buf285, reinterpret_tensor(primals_150, (384, 192), (1, 384), 0), alpha=1, beta=1, out=buf286)
    del primals_151
    buf287 = buf279; del buf279  # reuse
    buf288 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf290 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    buf291 = empty((2048, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_40(c_void_p(buf263.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    del primals_153
    buf292 = reinterpret_tensor(buf137, (2048, 576), (576, 1), 0); del buf137  # reuse
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_155, buf291, reinterpret_tensor(primals_154, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf292)
    del primals_155
    # Source Nodes: [x_235], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf293 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf292, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf292, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf292, (32, 4, 64, 48), (36864, 48, 576, 1), 384))
    buf294 = buf293[0]
    buf295 = buf293[1]
    buf296 = buf293[2]
    buf297 = buf293[3]
    buf298 = buf293[6]
    buf299 = buf293[7]
    del buf293
    buf301 = empty((2048, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_237], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_157, reinterpret_tensor(buf294, (2048, 192), (192, 1), 0), reinterpret_tensor(primals_156, (192, 192), (1, 192), 0), alpha=1, beta=1, out=buf301)
    del primals_157
    buf302 = buf287; del buf287  # reuse
    buf303 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf305 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    buf306 = empty((2048, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_41(c_void_p(buf263.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del primals_159
    buf307 = empty((2048, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_240], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_161, buf306, reinterpret_tensor(primals_160, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf307)
    del primals_161
    buf308 = empty((2048, 384), device='cpu', dtype=torch.float32)
    cpp_fused_silu_view_42(c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    buf309 = empty((2048, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_244], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_163, buf308, reinterpret_tensor(primals_162, (384, 192), (1, 384), 0), alpha=1, beta=1, out=buf309)
    del primals_163
    buf310 = reinterpret_tensor(buf309, (32, 64, 192), (12288, 192, 1), 0); del buf309  # reuse
    buf311 = buf302; del buf302  # reuse
    buf312 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf314 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    buf315 = empty_strided((8, 192, 16, 16), (49152, 1, 3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_43(c_void_p(buf310.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()))
    del buf263
    del primals_165
    # Source Nodes: [x_252], Original ATen: [aten.convolution]
    buf316 = extern_kernels.convolution(buf315, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf316, (8, 128, 16, 16), (32768, 1, 2048, 128))
    buf317 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf318 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf320 = empty((128, ), device='cpu', dtype=torch.float32)
    buf321 = empty_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    buf322 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_cat_44(c_void_p(buf316.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()))
    del primals_48
    # Source Nodes: [x_258], Original ATen: [aten.convolution]
    buf323 = extern_kernels.convolution(buf322, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf323, (8, 128, 16, 16), (32768, 1, 2048, 128))
    buf324 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf325 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf327 = empty((128, ), device='cpu', dtype=torch.float32)
    buf328 = empty_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    buf329 = empty_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_45(c_void_p(buf323.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()))
    del primals_50
    # Source Nodes: [x_264], Original ATen: [aten.convolution]
    buf330 = extern_kernels.convolution(buf329, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf330, (8, 512, 16, 16), (131072, 1, 8192, 512))
    buf331 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf332 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf334 = empty((512, ), device='cpu', dtype=torch.float32)
    buf335 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    buf336 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_46(c_void_p(buf330.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    del primals_52
    # Source Nodes: [x_270], Original ATen: [aten.convolution]
    buf337 = extern_kernels.convolution(buf336, primals_169, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
    assert_size_stride(buf337, (8, 512, 8, 8), (32768, 1, 4096, 512))
    buf338 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf339 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf341 = empty((512, ), device='cpu', dtype=torch.float32)
    buf342 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    buf343 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_47(c_void_p(buf337.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    del primals_54
    # Source Nodes: [x_278], Original ATen: [aten.convolution]
    buf344 = extern_kernels.convolution(buf343, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf344, (8, 160, 8, 8), (10240, 1, 1280, 160))
    buf345 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf346 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf348 = empty((160, ), device='cpu', dtype=torch.float32)
    buf349 = empty_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_48(c_void_p(buf344.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()))
    del primals_56
    # Source Nodes: [x_286], Original ATen: [aten.convolution]
    buf350 = extern_kernels.convolution(buf349, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf350, (8, 160, 8, 8), (10240, 1, 1280, 160))
    buf351 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf352 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf354 = empty((160, ), device='cpu', dtype=torch.float32)
    buf355 = empty_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    buf356 = empty_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_49(c_void_p(buf350.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()))
    del primals_58
    # Source Nodes: [x_292], Original ATen: [aten.convolution]
    buf357 = extern_kernels.convolution(buf356, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf357, (8, 240, 8, 8), (15360, 1, 1920, 240))
    buf358 = empty_strided((32, 16, 1), (1, 32, 512), device='cpu', dtype=torch.float32)
    buf359 = empty_strided((32, 16, 1), (1, 32, 512), device='cpu', dtype=torch.float32)
    buf361 = empty((32, 16, 240), device='cpu', dtype=torch.float32)
    buf362 = empty((512, 240), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_50(c_void_p(buf357.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    del primals_174
    buf363 = empty((512, 720), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_176, buf362, reinterpret_tensor(primals_175, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf363)
    del primals_176
    # Source Nodes: [x_295], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf364 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf363, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf363, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf363, (32, 4, 16, 60), (11520, 60, 720, 1), 480))
    buf365 = buf364[0]
    buf366 = buf364[1]
    buf367 = buf364[2]
    buf368 = buf364[3]
    buf369 = buf364[6]
    buf370 = buf364[7]
    del buf364
    buf372 = empty((512, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_297], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_178, reinterpret_tensor(buf365, (512, 240), (240, 1), 0), reinterpret_tensor(primals_177, (240, 240), (1, 240), 0), alpha=1, beta=1, out=buf372)
    del primals_178
    buf373 = reinterpret_tensor(buf358, (32, 16, 1), (16, 1, 512), 0); del buf358  # reuse
    buf374 = empty_strided((32, 16, 1), (16, 1, 512), device='cpu', dtype=torch.float32)
    buf376 = empty((32, 16, 240), device='cpu', dtype=torch.float32)
    buf377 = empty((512, 240), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_51(c_void_p(buf357.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del primals_180
    buf378 = empty((512, 480), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_300], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_182, buf377, reinterpret_tensor(primals_181, (240, 480), (1, 240), 0), alpha=1, beta=1, out=buf378)
    del primals_182
    buf379 = empty((512, 480), device='cpu', dtype=torch.float32)
    cpp_fused_silu_view_52(c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()))
    buf380 = empty((512, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_304], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_184, buf379, reinterpret_tensor(primals_183, (480, 240), (1, 480), 0), alpha=1, beta=1, out=buf380)
    del primals_184
    buf381 = buf373; del buf373  # reuse
    buf382 = empty_strided((32, 16, 1), (16, 1, 512), device='cpu', dtype=torch.float32)
    buf384 = empty((32, 16, 240), device='cpu', dtype=torch.float32)
    buf385 = empty((512, 240), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_53(c_void_p(buf357.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()))
    del primals_186
    buf386 = empty((512, 720), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_188, buf385, reinterpret_tensor(primals_187, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf386)
    del primals_188
    # Source Nodes: [x_307], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf387 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf386, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf386, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf386, (32, 4, 16, 60), (11520, 60, 720, 1), 480))
    buf388 = buf387[0]
    buf389 = buf387[1]
    buf390 = buf387[2]
    buf391 = buf387[3]
    buf392 = buf387[6]
    buf393 = buf387[7]
    del buf387
    buf395 = empty((512, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_309], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_190, reinterpret_tensor(buf388, (512, 240), (240, 1), 0), reinterpret_tensor(primals_189, (240, 240), (1, 240), 0), alpha=1, beta=1, out=buf395)
    del primals_190
    buf396 = buf381; del buf381  # reuse
    buf397 = empty_strided((32, 16, 1), (16, 1, 512), device='cpu', dtype=torch.float32)
    buf399 = empty((32, 16, 240), device='cpu', dtype=torch.float32)
    buf400 = empty((512, 240), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_54(c_void_p(buf357.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()))
    del primals_192
    buf401 = empty((512, 480), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_312], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_194, buf400, reinterpret_tensor(primals_193, (240, 480), (1, 240), 0), alpha=1, beta=1, out=buf401)
    del primals_194
    buf402 = empty((512, 480), device='cpu', dtype=torch.float32)
    cpp_fused_silu_view_55(c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()))
    buf403 = empty((512, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_316], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_196, buf402, reinterpret_tensor(primals_195, (480, 240), (1, 480), 0), alpha=1, beta=1, out=buf403)
    del primals_196
    buf404 = reinterpret_tensor(buf403, (32, 16, 240), (3840, 240, 1), 0); del buf403  # reuse
    buf405 = buf396; del buf396  # reuse
    buf406 = empty_strided((32, 16, 1), (16, 1, 512), device='cpu', dtype=torch.float32)
    buf408 = empty((32, 16, 240), device='cpu', dtype=torch.float32)
    buf409 = empty((512, 240), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_56(c_void_p(buf404.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    del primals_198
    buf410 = empty((512, 720), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_200, buf409, reinterpret_tensor(primals_199, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf410)
    del primals_200
    # Source Nodes: [x_319], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf411 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf410, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf410, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf410, (32, 4, 16, 60), (11520, 60, 720, 1), 480))
    buf412 = buf411[0]
    buf413 = buf411[1]
    buf414 = buf411[2]
    buf415 = buf411[3]
    buf416 = buf411[6]
    buf417 = buf411[7]
    del buf411
    buf419 = buf395; del buf395  # reuse
    # Source Nodes: [x_321], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_202, reinterpret_tensor(buf412, (512, 240), (240, 1), 0), reinterpret_tensor(primals_201, (240, 240), (1, 240), 0), alpha=1, beta=1, out=buf419)
    del primals_202
    buf420 = buf405; del buf405  # reuse
    buf421 = empty_strided((32, 16, 1), (16, 1, 512), device='cpu', dtype=torch.float32)
    buf423 = reinterpret_tensor(buf380, (32, 16, 240), (3840, 240, 1), 0); del buf380  # reuse
    buf424 = buf372; del buf372  # reuse
    cpp_fused_add_native_layer_norm_view_57(c_void_p(buf404.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()))
    del primals_204
    buf425 = empty((512, 480), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_324], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_206, buf424, reinterpret_tensor(primals_205, (240, 480), (1, 240), 0), alpha=1, beta=1, out=buf425)
    del primals_206
    buf426 = empty((512, 480), device='cpu', dtype=torch.float32)
    cpp_fused_silu_view_58(c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()))
    buf427 = reinterpret_tensor(buf357, (512, 240), (240, 1), 0); del buf357  # reuse
    # Source Nodes: [x_328], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_208, buf426, reinterpret_tensor(primals_207, (480, 240), (1, 480), 0), alpha=1, beta=1, out=buf427)
    del primals_208
    buf428 = buf420; del buf420  # reuse
    buf429 = empty_strided((32, 16, 1), (16, 1, 512), device='cpu', dtype=torch.float32)
    buf431 = empty((32, 16, 240), device='cpu', dtype=torch.float32)
    buf432 = empty_strided((8, 240, 8, 8), (15360, 1, 1920, 240), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_59(c_void_p(buf404.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()))
    del primals_210
    # Source Nodes: [x_336], Original ATen: [aten.convolution]
    buf433 = extern_kernels.convolution(buf432, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf433, (8, 160, 8, 8), (10240, 1, 1280, 160))
    buf434 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf435 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf437 = empty((160, ), device='cpu', dtype=torch.float32)
    buf438 = empty_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    buf439 = empty_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_cat_60(c_void_p(buf433.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()))
    del primals_60
    # Source Nodes: [x_342], Original ATen: [aten.convolution]
    buf440 = extern_kernels.convolution(buf439, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf440, (8, 160, 8, 8), (10240, 1, 1280, 160))
    buf441 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf442 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf444 = empty((160, ), device='cpu', dtype=torch.float32)
    buf445 = empty_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    buf446 = empty_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    buf457 = empty_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_61(c_void_p(buf440.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf457.data_ptr()))
    del buf445
    del primals_62
    # Source Nodes: [x_349], Original ATen: [aten.convolution]
    buf447 = extern_kernels.convolution(buf446, primals_213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf447, (8, 640, 8, 8), (40960, 1, 5120, 640))
    buf448 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf449 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cpu', dtype=torch.float32)
    buf451 = empty((640, ), device='cpu', dtype=torch.float32)
    buf452 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    buf453 = empty_strided((8, 640, 1, 1), (640, 1, 5120, 5120), device='cpu', dtype=torch.float32)
    buf454 = reinterpret_tensor(buf453, (8, 640), (640, 1), 0); del buf453  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_silu_view_62(c_void_p(buf454.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()))
    del primals_64
    buf455 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_360], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_215, buf454, reinterpret_tensor(primals_214, (640, 1000), (1, 640), 0), alpha=1, beta=1, out=buf455)
    del primals_215
    buf456 = buf452; del buf452  # reuse
    buf458 = buf438; del buf438  # reuse
    buf459 = reinterpret_tensor(buf429, (32, 16, 1), (16, 1, 1), 0); del buf429  # reuse
    buf460 = reinterpret_tensor(buf421, (32, 16, 1), (16, 1, 1), 0); del buf421  # reuse
    buf461 = reinterpret_tensor(buf427, (32, 4, 16, 60), (3840, 1, 240, 4), 0); del buf427  # reuse
    buf462 = reinterpret_tensor(buf406, (32, 16, 1), (16, 1, 1), 0); del buf406  # reuse
    buf463 = reinterpret_tensor(buf397, (32, 16, 1), (16, 1, 1), 0); del buf397  # reuse
    buf464 = reinterpret_tensor(buf419, (32, 4, 16, 60), (3840, 1, 240, 4), 0); del buf419  # reuse
    buf465 = reinterpret_tensor(buf382, (32, 16, 1), (16, 1, 1), 0); del buf382  # reuse
    buf466 = reinterpret_tensor(buf374, (32, 16, 1), (16, 1, 1), 0); del buf374  # reuse
    buf467 = reinterpret_tensor(buf404, (32, 4, 16, 60), (3840, 1, 240, 4), 0); del buf404  # reuse
    buf468 = reinterpret_tensor(buf428, (32, 16, 1), (16, 1, 1), 0); del buf428  # reuse
    buf469 = buf355; del buf355  # reuse
    buf470 = buf342; del buf342  # reuse
    buf471 = buf335; del buf335  # reuse
    buf472 = buf328; del buf328  # reuse
    buf473 = buf321; del buf321  # reuse
    buf474 = reinterpret_tensor(buf312, (32, 64, 1), (64, 1, 1), 0); del buf312  # reuse
    buf475 = reinterpret_tensor(buf303, (32, 64, 1), (64, 1, 1), 0); del buf303  # reuse
    buf476 = reinterpret_tensor(buf310, (32, 4, 64, 48), (12288, 1, 192, 4), 0); del buf310  # reuse
    buf477 = reinterpret_tensor(buf288, (32, 64, 1), (64, 1, 1), 0); del buf288  # reuse
    buf478 = reinterpret_tensor(buf280, (32, 64, 1), (64, 1, 1), 0); del buf280  # reuse
    buf479 = reinterpret_tensor(buf301, (32, 4, 64, 48), (12288, 1, 192, 4), 0); del buf301  # reuse
    buf480 = reinterpret_tensor(buf265, (32, 64, 1), (64, 1, 1), 0); del buf265  # reuse
    buf481 = reinterpret_tensor(buf256, (32, 64, 1), (64, 1, 1), 0); del buf256  # reuse
    buf482 = reinterpret_tensor(buf286, (32, 4, 64, 48), (12288, 1, 192, 4), 0); del buf286  # reuse
    buf483 = reinterpret_tensor(buf241, (32, 64, 1), (64, 1, 1), 0); del buf241  # reuse
    buf484 = reinterpret_tensor(buf233, (32, 64, 1), (64, 1, 1), 0); del buf233  # reuse
    buf485 = reinterpret_tensor(buf278, (32, 4, 64, 48), (12288, 1, 192, 4), 0); del buf278  # reuse
    buf486 = reinterpret_tensor(buf311, (32, 64, 1), (64, 1, 1), 0); del buf311  # reuse
    buf487 = buf214; del buf214  # reuse
    buf488 = buf201; del buf201  # reuse
    buf489 = buf194; del buf194  # reuse
    buf490 = buf187; del buf187  # reuse
    buf491 = buf180; del buf180  # reuse
    buf492 = reinterpret_tensor(buf171, (32, 256, 1), (256, 1, 1), 0); del buf171  # reuse
    buf493 = reinterpret_tensor(buf162, (32, 256, 1), (256, 1, 1), 0); del buf162  # reuse
    buf494 = reinterpret_tensor(buf122, (32, 4, 256, 36), (36864, 1, 144, 4), 0); del buf122  # reuse
    buf495 = reinterpret_tensor(buf147, (32, 256, 1), (256, 1, 1), 0); del buf147  # reuse
    buf496 = reinterpret_tensor(buf139, (32, 256, 1), (256, 1, 1), 0); del buf139  # reuse
    buf497 = empty_strided((32, 4, 256, 36), (36864, 1, 144, 4), device='cpu', dtype=torch.float32)
    buf498 = reinterpret_tensor(buf170, (32, 256, 1), (256, 1, 1), 0); del buf170  # reuse
    buf499 = buf120; del buf120  # reuse
    buf500 = buf107; del buf107  # reuse
    buf501 = buf100; del buf100  # reuse
    buf502 = buf87; del buf87  # reuse
    buf503 = buf80; del buf80  # reuse
    buf504 = buf67; del buf67  # reuse
    buf505 = buf60; del buf60  # reuse
    buf506 = buf47; del buf47  # reuse
    buf507 = buf40; del buf40  # reuse
    buf508 = buf27; del buf27  # reuse
    buf509 = buf20; del buf20  # reuse
    buf510 = buf13; del buf13  # reuse
    buf517 = reinterpret_tensor(buf10, (16, ), (1, ), 0); del buf10  # reuse
    buf525 = reinterpret_tensor(buf17, (64, ), (1, ), 0); del buf17  # reuse
    buf533 = reinterpret_tensor(buf24, (64, ), (1, ), 0); del buf24  # reuse
    buf541 = reinterpret_tensor(buf31, (32, ), (1, ), 0); del buf31  # reuse
    buf549 = reinterpret_tensor(buf37, (128, ), (1, ), 0); del buf37  # reuse
    buf557 = reinterpret_tensor(buf44, (128, ), (1, ), 0); del buf44  # reuse
    buf565 = reinterpret_tensor(buf51, (64, ), (1, ), 0); del buf51  # reuse
    buf573 = reinterpret_tensor(buf57, (256, ), (1, ), 0); del buf57  # reuse
    buf581 = reinterpret_tensor(buf64, (256, ), (1, ), 0); del buf64  # reuse
    buf589 = reinterpret_tensor(buf71, (64, ), (1, ), 0); del buf71  # reuse
    buf597 = reinterpret_tensor(buf77, (256, ), (1, ), 0); del buf77  # reuse
    buf605 = reinterpret_tensor(buf84, (256, ), (1, ), 0); del buf84  # reuse
    buf613 = reinterpret_tensor(buf91, (64, ), (1, ), 0); del buf91  # reuse
    buf621 = reinterpret_tensor(buf97, (256, ), (1, ), 0); del buf97  # reuse
    buf629 = reinterpret_tensor(buf104, (256, ), (1, ), 0); del buf104  # reuse
    buf637 = reinterpret_tensor(buf111, (96, ), (1, ), 0); del buf111  # reuse
    buf645 = reinterpret_tensor(buf117, (96, ), (1, ), 0); del buf117  # reuse
    buf653 = reinterpret_tensor(buf177, (96, ), (1, ), 0); del buf177  # reuse
    buf661 = reinterpret_tensor(buf184, (96, ), (1, ), 0); del buf184  # reuse
    buf669 = reinterpret_tensor(buf191, (384, ), (1, ), 0); del buf191  # reuse
    buf677 = reinterpret_tensor(buf198, (384, ), (1, ), 0); del buf198  # reuse
    buf685 = reinterpret_tensor(buf205, (128, ), (1, ), 0); del buf205  # reuse
    buf693 = reinterpret_tensor(buf211, (128, ), (1, ), 0); del buf211  # reuse
    buf701 = reinterpret_tensor(buf318, (128, ), (1, ), 0); del buf318  # reuse
    buf709 = reinterpret_tensor(buf325, (128, ), (1, ), 0); del buf325  # reuse
    buf717 = reinterpret_tensor(buf332, (512, ), (1, ), 0); del buf332  # reuse
    buf725 = reinterpret_tensor(buf339, (512, ), (1, ), 0); del buf339  # reuse
    buf733 = reinterpret_tensor(buf346, (160, ), (1, ), 0); del buf346  # reuse
    buf741 = reinterpret_tensor(buf352, (160, ), (1, ), 0); del buf352  # reuse
    buf749 = reinterpret_tensor(buf435, (160, ), (1, ), 0); del buf435  # reuse
    buf757 = reinterpret_tensor(buf442, (160, ), (1, ), 0); del buf442  # reuse
    buf765 = reinterpret_tensor(buf449, (640, ), (1, ), 0); del buf449  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_detach_fill_mul_native_layer_norm_native_layer_norm_backward_sigmoid_sub_63(c_void_p(buf456.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf749.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf765.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()))
    del buf124
    del buf218
    del buf359
    del buf517
    del buf525
    del buf533
    del buf541
    del buf549
    del buf557
    del buf565
    del buf573
    del buf581
    del buf589
    del buf597
    del buf605
    del buf613
    del buf621
    del buf629
    del buf637
    del buf645
    del buf653
    del buf661
    del buf669
    del buf677
    del buf685
    del buf693
    del buf701
    del buf709
    del buf717
    del buf725
    del buf733
    del buf741
    del buf749
    del buf757
    del buf765
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
    return (buf455, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, buf0, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, buf1, primals_82, primals_83, primals_89, primals_95, primals_101, primals_107, primals_109, buf2, primals_111, primals_112, primals_113, buf3, primals_115, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_152, primals_158, primals_164, primals_166, buf4, primals_168, primals_169, primals_170, buf5, primals_172, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_211, buf6, primals_213, buf7, buf8, buf12, buf14, buf15, buf19, buf21, buf22, buf26, buf28, buf29, buf33, buf34, buf35, buf39, buf41, buf42, buf46, buf48, buf49, buf53, buf54, buf55, buf59, buf61, buf62, buf66, buf68, buf69, buf73, buf74, buf75, buf79, buf81, buf82, buf86, buf88, buf89, buf93, buf94, buf95, buf99, buf101, buf102, buf106, buf108, buf109, buf113, buf114, buf115, buf119, buf121, buf126, buf127, reinterpret_tensor(buf128, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf128, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf128, (32, 4, 256, 36), (110592, 36, 432, 1), 288), buf131, buf132, buf133, buf134, buf135, reinterpret_tensor(buf130, (8192, 144), (144, 1), 0), buf141, buf142, buf143, buf144, buf149, buf150, reinterpret_tensor(buf151, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf151, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf151, (32, 4, 256, 36), (110592, 36, 432, 1), 288), buf154, buf155, buf156, buf157, buf158, reinterpret_tensor(buf153, (8192, 144), (144, 1), 0), buf164, buf165, buf166, buf167, buf173, buf174, buf175, buf179, buf181, buf182, buf186, buf188, buf189, buf193, buf195, buf196, buf200, buf202, buf203, buf207, buf208, buf209, buf213, buf215, buf220, buf221, reinterpret_tensor(buf222, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf222, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf222, (32, 4, 64, 48), (36864, 48, 576, 1), 384), buf225, buf226, buf227, buf228, buf229, reinterpret_tensor(buf224, (2048, 192), (192, 1), 0), buf235, buf236, buf237, buf238, buf243, buf244, reinterpret_tensor(buf245, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf245, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf245, (32, 4, 64, 48), (36864, 48, 576, 1), 384), buf248, buf249, buf250, buf251, buf252, reinterpret_tensor(buf247, (2048, 192), (192, 1), 0), buf258, buf259, buf260, buf261, buf267, buf268, reinterpret_tensor(buf269, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf269, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf269, (32, 4, 64, 48), (36864, 48, 576, 1), 384), buf272, buf273, buf274, buf275, buf276, reinterpret_tensor(buf271, (2048, 192), (192, 1), 0), buf282, buf283, buf284, buf285, buf290, buf291, reinterpret_tensor(buf292, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf292, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf292, (32, 4, 64, 48), (36864, 48, 576, 1), 384), buf295, buf296, buf297, buf298, buf299, reinterpret_tensor(buf294, (2048, 192), (192, 1), 0), buf305, buf306, buf307, buf308, buf314, buf315, buf316, buf320, buf322, buf323, buf327, buf329, buf330, buf334, buf336, buf337, buf341, buf343, buf344, buf348, buf349, buf350, buf354, buf356, buf361, buf362, reinterpret_tensor(buf363, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf363, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf363, (32, 4, 16, 60), (11520, 60, 720, 1), 480), buf366, buf367, buf368, buf369, buf370, reinterpret_tensor(buf365, (512, 240), (240, 1), 0), buf376, buf377, buf378, buf379, buf384, buf385, reinterpret_tensor(buf386, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf386, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf386, (32, 4, 16, 60), (11520, 60, 720, 1), 480), buf389, buf390, buf391, buf392, buf393, reinterpret_tensor(buf388, (512, 240), (240, 1), 0), buf399, buf400, buf401, buf402, buf408, buf409, reinterpret_tensor(buf410, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf410, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf410, (32, 4, 16, 60), (11520, 60, 720, 1), 480), buf413, buf414, buf415, buf416, buf417, reinterpret_tensor(buf412, (512, 240), (240, 1), 0), buf423, buf424, buf425, buf426, buf431, buf432, buf433, buf437, buf439, buf440, buf444, buf446, buf447, buf451, buf454, reinterpret_tensor(primals_214, (1000, 640), (640, 1), 0), buf456, reinterpret_tensor(buf448, (1, 640, 1, 1), (640, 1, 1, 1), 0), buf457, reinterpret_tensor(buf441, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf458, reinterpret_tensor(buf434, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf459, reinterpret_tensor(primals_207, (240, 480), (480, 1), 0), reinterpret_tensor(primals_205, (480, 240), (240, 1), 0), buf460, reinterpret_tensor(primals_201, (240, 240), (240, 1), 0), buf461, reinterpret_tensor(primals_199, (720, 240), (240, 1), 0), buf462, reinterpret_tensor(primals_195, (240, 480), (480, 1), 0), reinterpret_tensor(primals_193, (480, 240), (240, 1), 0), buf463, reinterpret_tensor(primals_189, (240, 240), (240, 1), 0), buf464, reinterpret_tensor(primals_187, (720, 240), (240, 1), 0), buf465, reinterpret_tensor(primals_183, (240, 480), (480, 1), 0), reinterpret_tensor(primals_181, (480, 240), (240, 1), 0), buf466, reinterpret_tensor(primals_177, (240, 240), (240, 1), 0), buf467, reinterpret_tensor(primals_175, (720, 240), (240, 1), 0), buf468, buf469, reinterpret_tensor(buf351, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf345, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf470, reinterpret_tensor(buf338, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf471, reinterpret_tensor(buf331, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf472, reinterpret_tensor(buf324, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf473, reinterpret_tensor(buf317, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf474, reinterpret_tensor(primals_162, (192, 384), (384, 1), 0), reinterpret_tensor(primals_160, (384, 192), (192, 1), 0), buf475, reinterpret_tensor(primals_156, (192, 192), (192, 1), 0), buf476, reinterpret_tensor(primals_154, (576, 192), (192, 1), 0), buf477, reinterpret_tensor(primals_150, (192, 384), (384, 1), 0), reinterpret_tensor(primals_148, (384, 192), (192, 1), 0), buf478, reinterpret_tensor(primals_144, (192, 192), (192, 1), 0), buf479, reinterpret_tensor(primals_142, (576, 192), (192, 1), 0), buf480, reinterpret_tensor(primals_138, (192, 384), (384, 1), 0), reinterpret_tensor(primals_136, (384, 192), (192, 1), 0), buf481, reinterpret_tensor(primals_132, (192, 192), (192, 1), 0), buf482, reinterpret_tensor(primals_130, (576, 192), (192, 1), 0), buf483, reinterpret_tensor(primals_126, (192, 384), (384, 1), 0), reinterpret_tensor(primals_124, (384, 192), (192, 1), 0), buf484, reinterpret_tensor(primals_120, (192, 192), (192, 1), 0), buf485, reinterpret_tensor(primals_118, (576, 192), (192, 1), 0), buf486, buf487, reinterpret_tensor(buf210, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf204, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf488, reinterpret_tensor(buf197, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf489, reinterpret_tensor(buf190, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf490, reinterpret_tensor(buf183, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf491, reinterpret_tensor(buf176, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf492, reinterpret_tensor(primals_105, (144, 288), (288, 1), 0), reinterpret_tensor(primals_103, (288, 144), (144, 1), 0), buf493, reinterpret_tensor(primals_99, (144, 144), (144, 1), 0), buf494, reinterpret_tensor(primals_97, (432, 144), (144, 1), 0), buf495, reinterpret_tensor(primals_93, (144, 288), (288, 1), 0), reinterpret_tensor(primals_91, (288, 144), (144, 1), 0), buf496, reinterpret_tensor(primals_87, (144, 144), (144, 1), 0), buf497, reinterpret_tensor(primals_85, (432, 144), (144, 1), 0), buf498, buf499, reinterpret_tensor(buf116, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf110, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf500, reinterpret_tensor(buf103, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf501, reinterpret_tensor(buf96, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf90, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf502, reinterpret_tensor(buf83, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf503, reinterpret_tensor(buf76, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf70, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf504, reinterpret_tensor(buf63, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf505, reinterpret_tensor(buf56, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf50, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf506, reinterpret_tensor(buf43, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf507, reinterpret_tensor(buf36, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf30, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf508, reinterpret_tensor(buf23, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf509, reinterpret_tensor(buf16, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf510, reinterpret_tensor(buf9, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((96, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((144, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((432, 144), (144, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((144, 144), (144, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((288, 144), (144, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((144, 288), (288, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((432, 144), (144, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((144, 144), (144, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((288, 144), (144, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((288, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((144, 288), (288, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((96, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((96, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((160, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((240, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((720, 240), (240, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((240, 240), (240, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((480, 240), (240, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((240, 480), (480, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((720, 240), (240, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((240, 240), (240, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((480, 240), (240, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((240, 480), (480, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((720, 240), (240, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((240, 240), (240, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((480, 240), (240, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((240, 480), (480, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((160, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((160, 320, 3, 3), (2880, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((1000, 640), (640, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_217 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_220 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_223 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_226 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_229 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_232 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_235 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_238 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_241 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_244 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_247 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_250 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_253 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_256 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_259 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_262 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_265 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_268 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_271 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_274 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_277 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_280 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_283 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_286 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_289 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_292 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_295 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_298 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_301 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_304 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_307 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_310 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilevit_s', benchmark_compiled_module)
