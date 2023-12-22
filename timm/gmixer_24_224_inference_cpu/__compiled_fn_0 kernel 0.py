
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


cpp_fused_convolution_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (256L*x1) + (768L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_4 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_8 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_9 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_12 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_16 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_17 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_20 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_24 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_25 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_28 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_32 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_33 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_36 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_40 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_41 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_44 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_48 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_49 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_52 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_56 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_57 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_60 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_64 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_65 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_68 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_72 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_73 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_76 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_80 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_81 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_84 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_88 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_89 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(384.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_92 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp16 = tmp15.rsqrt();
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp20 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp2 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp8 = tmp6 + tmp7;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp13 = static_cast<float>(384.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(384.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_96 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_native_layer_norm_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
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
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp8;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                            auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(384.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-06);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1 = args
    args.clear()
    assert_size_stride(arg0_1, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg1_1, (384, ), (1, ))
    assert_size_stride(arg2_1, (384, ), (1, ))
    assert_size_stride(arg3_1, (384, ), (1, ))
    assert_size_stride(arg4_1, (384, 196), (196, 1))
    assert_size_stride(arg5_1, (384, ), (1, ))
    assert_size_stride(arg6_1, (196, 192), (192, 1))
    assert_size_stride(arg7_1, (196, ), (1, ))
    assert_size_stride(arg8_1, (384, ), (1, ))
    assert_size_stride(arg9_1, (384, ), (1, ))
    assert_size_stride(arg10_1, (1536, 384), (384, 1))
    assert_size_stride(arg11_1, (1536, ), (1, ))
    assert_size_stride(arg12_1, (384, 768), (768, 1))
    assert_size_stride(arg13_1, (384, ), (1, ))
    assert_size_stride(arg14_1, (384, ), (1, ))
    assert_size_stride(arg15_1, (384, ), (1, ))
    assert_size_stride(arg16_1, (384, 196), (196, 1))
    assert_size_stride(arg17_1, (384, ), (1, ))
    assert_size_stride(arg18_1, (196, 192), (192, 1))
    assert_size_stride(arg19_1, (196, ), (1, ))
    assert_size_stride(arg20_1, (384, ), (1, ))
    assert_size_stride(arg21_1, (384, ), (1, ))
    assert_size_stride(arg22_1, (1536, 384), (384, 1))
    assert_size_stride(arg23_1, (1536, ), (1, ))
    assert_size_stride(arg24_1, (384, 768), (768, 1))
    assert_size_stride(arg25_1, (384, ), (1, ))
    assert_size_stride(arg26_1, (384, ), (1, ))
    assert_size_stride(arg27_1, (384, ), (1, ))
    assert_size_stride(arg28_1, (384, 196), (196, 1))
    assert_size_stride(arg29_1, (384, ), (1, ))
    assert_size_stride(arg30_1, (196, 192), (192, 1))
    assert_size_stride(arg31_1, (196, ), (1, ))
    assert_size_stride(arg32_1, (384, ), (1, ))
    assert_size_stride(arg33_1, (384, ), (1, ))
    assert_size_stride(arg34_1, (1536, 384), (384, 1))
    assert_size_stride(arg35_1, (1536, ), (1, ))
    assert_size_stride(arg36_1, (384, 768), (768, 1))
    assert_size_stride(arg37_1, (384, ), (1, ))
    assert_size_stride(arg38_1, (384, ), (1, ))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (384, 196), (196, 1))
    assert_size_stride(arg41_1, (384, ), (1, ))
    assert_size_stride(arg42_1, (196, 192), (192, 1))
    assert_size_stride(arg43_1, (196, ), (1, ))
    assert_size_stride(arg44_1, (384, ), (1, ))
    assert_size_stride(arg45_1, (384, ), (1, ))
    assert_size_stride(arg46_1, (1536, 384), (384, 1))
    assert_size_stride(arg47_1, (1536, ), (1, ))
    assert_size_stride(arg48_1, (384, 768), (768, 1))
    assert_size_stride(arg49_1, (384, ), (1, ))
    assert_size_stride(arg50_1, (384, ), (1, ))
    assert_size_stride(arg51_1, (384, ), (1, ))
    assert_size_stride(arg52_1, (384, 196), (196, 1))
    assert_size_stride(arg53_1, (384, ), (1, ))
    assert_size_stride(arg54_1, (196, 192), (192, 1))
    assert_size_stride(arg55_1, (196, ), (1, ))
    assert_size_stride(arg56_1, (384, ), (1, ))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (1536, 384), (384, 1))
    assert_size_stride(arg59_1, (1536, ), (1, ))
    assert_size_stride(arg60_1, (384, 768), (768, 1))
    assert_size_stride(arg61_1, (384, ), (1, ))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (384, 196), (196, 1))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (196, 192), (192, 1))
    assert_size_stride(arg67_1, (196, ), (1, ))
    assert_size_stride(arg68_1, (384, ), (1, ))
    assert_size_stride(arg69_1, (384, ), (1, ))
    assert_size_stride(arg70_1, (1536, 384), (384, 1))
    assert_size_stride(arg71_1, (1536, ), (1, ))
    assert_size_stride(arg72_1, (384, 768), (768, 1))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (384, 196), (196, 1))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (196, 192), (192, 1))
    assert_size_stride(arg79_1, (196, ), (1, ))
    assert_size_stride(arg80_1, (384, ), (1, ))
    assert_size_stride(arg81_1, (384, ), (1, ))
    assert_size_stride(arg82_1, (1536, 384), (384, 1))
    assert_size_stride(arg83_1, (1536, ), (1, ))
    assert_size_stride(arg84_1, (384, 768), (768, 1))
    assert_size_stride(arg85_1, (384, ), (1, ))
    assert_size_stride(arg86_1, (384, ), (1, ))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (384, 196), (196, 1))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (196, 192), (192, 1))
    assert_size_stride(arg91_1, (196, ), (1, ))
    assert_size_stride(arg92_1, (384, ), (1, ))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (1536, 384), (384, 1))
    assert_size_stride(arg95_1, (1536, ), (1, ))
    assert_size_stride(arg96_1, (384, 768), (768, 1))
    assert_size_stride(arg97_1, (384, ), (1, ))
    assert_size_stride(arg98_1, (384, ), (1, ))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (384, 196), (196, 1))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (196, 192), (192, 1))
    assert_size_stride(arg103_1, (196, ), (1, ))
    assert_size_stride(arg104_1, (384, ), (1, ))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (1536, 384), (384, 1))
    assert_size_stride(arg107_1, (1536, ), (1, ))
    assert_size_stride(arg108_1, (384, 768), (768, 1))
    assert_size_stride(arg109_1, (384, ), (1, ))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, 196), (196, 1))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (196, 192), (192, 1))
    assert_size_stride(arg115_1, (196, ), (1, ))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (1536, 384), (384, 1))
    assert_size_stride(arg119_1, (1536, ), (1, ))
    assert_size_stride(arg120_1, (384, 768), (768, 1))
    assert_size_stride(arg121_1, (384, ), (1, ))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (384, 196), (196, 1))
    assert_size_stride(arg125_1, (384, ), (1, ))
    assert_size_stride(arg126_1, (196, 192), (192, 1))
    assert_size_stride(arg127_1, (196, ), (1, ))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (384, ), (1, ))
    assert_size_stride(arg130_1, (1536, 384), (384, 1))
    assert_size_stride(arg131_1, (1536, ), (1, ))
    assert_size_stride(arg132_1, (384, 768), (768, 1))
    assert_size_stride(arg133_1, (384, ), (1, ))
    assert_size_stride(arg134_1, (384, ), (1, ))
    assert_size_stride(arg135_1, (384, ), (1, ))
    assert_size_stride(arg136_1, (384, 196), (196, 1))
    assert_size_stride(arg137_1, (384, ), (1, ))
    assert_size_stride(arg138_1, (196, 192), (192, 1))
    assert_size_stride(arg139_1, (196, ), (1, ))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (384, ), (1, ))
    assert_size_stride(arg142_1, (1536, 384), (384, 1))
    assert_size_stride(arg143_1, (1536, ), (1, ))
    assert_size_stride(arg144_1, (384, 768), (768, 1))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (384, ), (1, ))
    assert_size_stride(arg148_1, (384, 196), (196, 1))
    assert_size_stride(arg149_1, (384, ), (1, ))
    assert_size_stride(arg150_1, (196, 192), (192, 1))
    assert_size_stride(arg151_1, (196, ), (1, ))
    assert_size_stride(arg152_1, (384, ), (1, ))
    assert_size_stride(arg153_1, (384, ), (1, ))
    assert_size_stride(arg154_1, (1536, 384), (384, 1))
    assert_size_stride(arg155_1, (1536, ), (1, ))
    assert_size_stride(arg156_1, (384, 768), (768, 1))
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (384, 196), (196, 1))
    assert_size_stride(arg161_1, (384, ), (1, ))
    assert_size_stride(arg162_1, (196, 192), (192, 1))
    assert_size_stride(arg163_1, (196, ), (1, ))
    assert_size_stride(arg164_1, (384, ), (1, ))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (1536, 384), (384, 1))
    assert_size_stride(arg167_1, (1536, ), (1, ))
    assert_size_stride(arg168_1, (384, 768), (768, 1))
    assert_size_stride(arg169_1, (384, ), (1, ))
    assert_size_stride(arg170_1, (384, ), (1, ))
    assert_size_stride(arg171_1, (384, ), (1, ))
    assert_size_stride(arg172_1, (384, 196), (196, 1))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (196, 192), (192, 1))
    assert_size_stride(arg175_1, (196, ), (1, ))
    assert_size_stride(arg176_1, (384, ), (1, ))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (1536, 384), (384, 1))
    assert_size_stride(arg179_1, (1536, ), (1, ))
    assert_size_stride(arg180_1, (384, 768), (768, 1))
    assert_size_stride(arg181_1, (384, ), (1, ))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (384, ), (1, ))
    assert_size_stride(arg184_1, (384, 196), (196, 1))
    assert_size_stride(arg185_1, (384, ), (1, ))
    assert_size_stride(arg186_1, (196, 192), (192, 1))
    assert_size_stride(arg187_1, (196, ), (1, ))
    assert_size_stride(arg188_1, (384, ), (1, ))
    assert_size_stride(arg189_1, (384, ), (1, ))
    assert_size_stride(arg190_1, (1536, 384), (384, 1))
    assert_size_stride(arg191_1, (1536, ), (1, ))
    assert_size_stride(arg192_1, (384, 768), (768, 1))
    assert_size_stride(arg193_1, (384, ), (1, ))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (384, ), (1, ))
    assert_size_stride(arg196_1, (384, 196), (196, 1))
    assert_size_stride(arg197_1, (384, ), (1, ))
    assert_size_stride(arg198_1, (196, 192), (192, 1))
    assert_size_stride(arg199_1, (196, ), (1, ))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (384, ), (1, ))
    assert_size_stride(arg202_1, (1536, 384), (384, 1))
    assert_size_stride(arg203_1, (1536, ), (1, ))
    assert_size_stride(arg204_1, (384, 768), (768, 1))
    assert_size_stride(arg205_1, (384, ), (1, ))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (384, ), (1, ))
    assert_size_stride(arg208_1, (384, 196), (196, 1))
    assert_size_stride(arg209_1, (384, ), (1, ))
    assert_size_stride(arg210_1, (196, 192), (192, 1))
    assert_size_stride(arg211_1, (196, ), (1, ))
    assert_size_stride(arg212_1, (384, ), (1, ))
    assert_size_stride(arg213_1, (384, ), (1, ))
    assert_size_stride(arg214_1, (1536, 384), (384, 1))
    assert_size_stride(arg215_1, (1536, ), (1, ))
    assert_size_stride(arg216_1, (384, 768), (768, 1))
    assert_size_stride(arg217_1, (384, ), (1, ))
    assert_size_stride(arg218_1, (384, ), (1, ))
    assert_size_stride(arg219_1, (384, ), (1, ))
    assert_size_stride(arg220_1, (384, 196), (196, 1))
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (196, 192), (192, 1))
    assert_size_stride(arg223_1, (196, ), (1, ))
    assert_size_stride(arg224_1, (384, ), (1, ))
    assert_size_stride(arg225_1, (384, ), (1, ))
    assert_size_stride(arg226_1, (1536, 384), (384, 1))
    assert_size_stride(arg227_1, (1536, ), (1, ))
    assert_size_stride(arg228_1, (384, 768), (768, 1))
    assert_size_stride(arg229_1, (384, ), (1, ))
    assert_size_stride(arg230_1, (384, ), (1, ))
    assert_size_stride(arg231_1, (384, ), (1, ))
    assert_size_stride(arg232_1, (384, 196), (196, 1))
    assert_size_stride(arg233_1, (384, ), (1, ))
    assert_size_stride(arg234_1, (196, 192), (192, 1))
    assert_size_stride(arg235_1, (196, ), (1, ))
    assert_size_stride(arg236_1, (384, ), (1, ))
    assert_size_stride(arg237_1, (384, ), (1, ))
    assert_size_stride(arg238_1, (1536, 384), (384, 1))
    assert_size_stride(arg239_1, (1536, ), (1, ))
    assert_size_stride(arg240_1, (384, 768), (768, 1))
    assert_size_stride(arg241_1, (384, ), (1, ))
    assert_size_stride(arg242_1, (384, ), (1, ))
    assert_size_stride(arg243_1, (384, ), (1, ))
    assert_size_stride(arg244_1, (384, 196), (196, 1))
    assert_size_stride(arg245_1, (384, ), (1, ))
    assert_size_stride(arg246_1, (196, 192), (192, 1))
    assert_size_stride(arg247_1, (196, ), (1, ))
    assert_size_stride(arg248_1, (384, ), (1, ))
    assert_size_stride(arg249_1, (384, ), (1, ))
    assert_size_stride(arg250_1, (1536, 384), (384, 1))
    assert_size_stride(arg251_1, (1536, ), (1, ))
    assert_size_stride(arg252_1, (384, 768), (768, 1))
    assert_size_stride(arg253_1, (384, ), (1, ))
    assert_size_stride(arg254_1, (384, ), (1, ))
    assert_size_stride(arg255_1, (384, ), (1, ))
    assert_size_stride(arg256_1, (384, 196), (196, 1))
    assert_size_stride(arg257_1, (384, ), (1, ))
    assert_size_stride(arg258_1, (196, 192), (192, 1))
    assert_size_stride(arg259_1, (196, ), (1, ))
    assert_size_stride(arg260_1, (384, ), (1, ))
    assert_size_stride(arg261_1, (384, ), (1, ))
    assert_size_stride(arg262_1, (1536, 384), (384, 1))
    assert_size_stride(arg263_1, (1536, ), (1, ))
    assert_size_stride(arg264_1, (384, 768), (768, 1))
    assert_size_stride(arg265_1, (384, ), (1, ))
    assert_size_stride(arg266_1, (384, ), (1, ))
    assert_size_stride(arg267_1, (384, ), (1, ))
    assert_size_stride(arg268_1, (384, 196), (196, 1))
    assert_size_stride(arg269_1, (384, ), (1, ))
    assert_size_stride(arg270_1, (196, 192), (192, 1))
    assert_size_stride(arg271_1, (196, ), (1, ))
    assert_size_stride(arg272_1, (384, ), (1, ))
    assert_size_stride(arg273_1, (384, ), (1, ))
    assert_size_stride(arg274_1, (1536, 384), (384, 1))
    assert_size_stride(arg275_1, (1536, ), (1, ))
    assert_size_stride(arg276_1, (384, 768), (768, 1))
    assert_size_stride(arg277_1, (384, ), (1, ))
    assert_size_stride(arg278_1, (384, ), (1, ))
    assert_size_stride(arg279_1, (384, ), (1, ))
    assert_size_stride(arg280_1, (384, 196), (196, 1))
    assert_size_stride(arg281_1, (384, ), (1, ))
    assert_size_stride(arg282_1, (196, 192), (192, 1))
    assert_size_stride(arg283_1, (196, ), (1, ))
    assert_size_stride(arg284_1, (384, ), (1, ))
    assert_size_stride(arg285_1, (384, ), (1, ))
    assert_size_stride(arg286_1, (1536, 384), (384, 1))
    assert_size_stride(arg287_1, (1536, ), (1, ))
    assert_size_stride(arg288_1, (384, 768), (768, 1))
    assert_size_stride(arg289_1, (384, ), (1, ))
    assert_size_stride(arg290_1, (384, ), (1, ))
    assert_size_stride(arg291_1, (384, ), (1, ))
    assert_size_stride(arg292_1, (1000, 384), (384, 1))
    assert_size_stride(arg293_1, (1000, ), (1, ))
    assert_size_stride(arg294_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((384, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg294_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg294_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg1_1
    del buf1
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 384, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg2_1
    del arg3_1
    buf7 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_4], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf6, (3072, 196), (196, 1), 0), reinterpret_tensor(arg4_1, (196, 384), (1, 196), 0), out=buf7)
    del arg4_1
    buf8 = empty((8, 384, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_2(c_void_p(buf7.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf8.data_ptr()))
    del arg5_1
    buf9 = reinterpret_tensor(buf6, (3072, 196), (196, 1), 0); del buf6  # reuse
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg7_1, reinterpret_tensor(buf8, (3072, 192), (192, 1), 0), reinterpret_tensor(arg6_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf9)
    del arg6_1
    del arg7_1
    buf10 = buf4; del buf4  # reuse
    buf11 = buf3; del buf3  # reuse
    buf13 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_3(c_void_p(buf2.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg8_1
    del arg9_1
    buf14 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg11_1, reinterpret_tensor(buf13, (1568, 384), (384, 1), 0), reinterpret_tensor(arg10_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf14)
    del arg10_1
    del arg11_1
    buf15 = reinterpret_tensor(buf0, (8, 196, 768), (150528, 768, 1), 0); del buf0  # reuse
    cpp_fused_mul_silu_4(c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    buf16 = reinterpret_tensor(buf13, (1568, 384), (384, 1), 0); del buf13  # reuse
    # Source Nodes: [x_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg13_1, reinterpret_tensor(buf15, (1568, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf16)
    del arg12_1
    del arg13_1
    buf17 = buf11; del buf11  # reuse
    buf18 = buf10; del buf10  # reuse
    buf20 = empty((8, 384, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_native_layer_norm_5(c_void_p(buf2.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()))
    del arg14_1
    del arg15_1
    buf21 = buf7; del buf7  # reuse
    # Source Nodes: [x_18], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (3072, 196), (196, 1), 0), reinterpret_tensor(arg16_1, (196, 384), (1, 196), 0), out=buf21)
    del arg16_1
    buf22 = buf8; del buf8  # reuse
    cpp_fused_mul_silu_6(c_void_p(buf21.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf22.data_ptr()))
    del arg17_1
    buf23 = reinterpret_tensor(buf20, (3072, 196), (196, 1), 0); del buf20  # reuse
    # Source Nodes: [x_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg19_1, reinterpret_tensor(buf22, (3072, 192), (192, 1), 0), reinterpret_tensor(arg18_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf23)
    del arg18_1
    del arg19_1
    buf24 = buf18; del buf18  # reuse
    buf25 = buf17; del buf17  # reuse
    buf27 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_7(c_void_p(buf2.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()))
    del arg20_1
    del arg21_1
    buf28 = buf14; del buf14  # reuse
    # Source Nodes: [x_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg23_1, reinterpret_tensor(buf27, (1568, 384), (384, 1), 0), reinterpret_tensor(arg22_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf28)
    del arg22_1
    del arg23_1
    buf29 = buf15; del buf15  # reuse
    cpp_fused_mul_silu_8(c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    buf30 = reinterpret_tensor(buf27, (1568, 384), (384, 1), 0); del buf27  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg25_1, reinterpret_tensor(buf29, (1568, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf30)
    del arg24_1
    del arg25_1
    buf31 = reinterpret_tensor(buf30, (8, 196, 384), (75264, 384, 1), 0); del buf30  # reuse
    buf32 = buf25; del buf25  # reuse
    buf33 = buf24; del buf24  # reuse
    buf35 = empty((8, 384, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_native_layer_norm_9(c_void_p(buf31.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf35.data_ptr()))
    del arg26_1
    del arg27_1
    buf36 = buf21; del buf21  # reuse
    # Source Nodes: [x_32], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (3072, 196), (196, 1), 0), reinterpret_tensor(arg28_1, (196, 384), (1, 196), 0), out=buf36)
    del arg28_1
    buf37 = buf22; del buf22  # reuse
    cpp_fused_mul_silu_10(c_void_p(buf36.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf37.data_ptr()))
    del arg29_1
    buf38 = reinterpret_tensor(buf35, (3072, 196), (196, 1), 0); del buf35  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg31_1, reinterpret_tensor(buf37, (3072, 192), (192, 1), 0), reinterpret_tensor(arg30_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf38)
    del arg30_1
    del arg31_1
    buf39 = buf33; del buf33  # reuse
    buf40 = buf32; del buf32  # reuse
    buf42 = reinterpret_tensor(buf9, (8, 196, 384), (75264, 384, 1), 0); del buf9  # reuse
    cpp_fused_add_native_layer_norm_11(c_void_p(buf31.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()))
    del arg32_1
    del arg33_1
    buf43 = buf28; del buf28  # reuse
    # Source Nodes: [x_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg35_1, reinterpret_tensor(buf42, (1568, 384), (384, 1), 0), reinterpret_tensor(arg34_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf43)
    del arg34_1
    del arg35_1
    buf44 = buf29; del buf29  # reuse
    cpp_fused_mul_silu_12(c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    buf45 = reinterpret_tensor(buf42, (1568, 384), (384, 1), 0); del buf42  # reuse
    # Source Nodes: [x_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg37_1, reinterpret_tensor(buf44, (1568, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf45)
    del arg36_1
    del arg37_1
    buf46 = buf40; del buf40  # reuse
    buf47 = buf39; del buf39  # reuse
    buf49 = reinterpret_tensor(buf23, (8, 384, 196), (75264, 196, 1), 0); del buf23  # reuse
    cpp_fused_add_clone_native_layer_norm_13(c_void_p(buf31.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()))
    del arg38_1
    del arg39_1
    buf50 = buf36; del buf36  # reuse
    # Source Nodes: [x_46], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (3072, 196), (196, 1), 0), reinterpret_tensor(arg40_1, (196, 384), (1, 196), 0), out=buf50)
    del arg40_1
    buf51 = buf37; del buf37  # reuse
    cpp_fused_mul_silu_14(c_void_p(buf50.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf51.data_ptr()))
    del arg41_1
    buf52 = reinterpret_tensor(buf49, (3072, 196), (196, 1), 0); del buf49  # reuse
    # Source Nodes: [x_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg43_1, reinterpret_tensor(buf51, (3072, 192), (192, 1), 0), reinterpret_tensor(arg42_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf52)
    del arg42_1
    del arg43_1
    buf53 = buf47; del buf47  # reuse
    buf54 = buf46; del buf46  # reuse
    buf56 = reinterpret_tensor(buf2, (8, 196, 384), (75264, 384, 1), 0); del buf2  # reuse
    cpp_fused_add_native_layer_norm_15(c_void_p(buf31.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()))
    del arg44_1
    del arg45_1
    buf57 = buf43; del buf43  # reuse
    # Source Nodes: [x_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg47_1, reinterpret_tensor(buf56, (1568, 384), (384, 1), 0), reinterpret_tensor(arg46_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf57)
    del arg46_1
    del arg47_1
    buf58 = buf44; del buf44  # reuse
    cpp_fused_mul_silu_16(c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    buf59 = reinterpret_tensor(buf56, (1568, 384), (384, 1), 0); del buf56  # reuse
    # Source Nodes: [x_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg49_1, reinterpret_tensor(buf58, (1568, 768), (768, 1), 0), reinterpret_tensor(arg48_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf59)
    del arg48_1
    del arg49_1
    buf60 = reinterpret_tensor(buf59, (8, 196, 384), (75264, 384, 1), 0); del buf59  # reuse
    buf61 = buf54; del buf54  # reuse
    buf62 = buf53; del buf53  # reuse
    buf64 = reinterpret_tensor(buf16, (8, 384, 196), (75264, 196, 1), 0); del buf16  # reuse
    cpp_fused_add_clone_native_layer_norm_17(c_void_p(buf60.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()))
    del arg50_1
    del arg51_1
    buf65 = buf50; del buf50  # reuse
    # Source Nodes: [x_60], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (3072, 196), (196, 1), 0), reinterpret_tensor(arg52_1, (196, 384), (1, 196), 0), out=buf65)
    del arg52_1
    buf66 = buf51; del buf51  # reuse
    cpp_fused_mul_silu_18(c_void_p(buf65.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf66.data_ptr()))
    del arg53_1
    buf67 = reinterpret_tensor(buf64, (3072, 196), (196, 1), 0); del buf64  # reuse
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf66, (3072, 192), (192, 1), 0), reinterpret_tensor(arg54_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf67)
    del arg54_1
    del arg55_1
    buf68 = buf62; del buf62  # reuse
    buf69 = buf61; del buf61  # reuse
    buf71 = reinterpret_tensor(buf52, (8, 196, 384), (75264, 384, 1), 0); del buf52  # reuse
    cpp_fused_add_native_layer_norm_19(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()))
    del arg56_1
    del arg57_1
    buf72 = buf57; del buf57  # reuse
    # Source Nodes: [x_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg59_1, reinterpret_tensor(buf71, (1568, 384), (384, 1), 0), reinterpret_tensor(arg58_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf72)
    del arg58_1
    del arg59_1
    buf73 = buf58; del buf58  # reuse
    cpp_fused_mul_silu_20(c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    buf74 = reinterpret_tensor(buf71, (1568, 384), (384, 1), 0); del buf71  # reuse
    # Source Nodes: [x_71], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg61_1, reinterpret_tensor(buf73, (1568, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf74)
    del arg60_1
    del arg61_1
    buf75 = buf69; del buf69  # reuse
    buf76 = buf68; del buf68  # reuse
    buf78 = reinterpret_tensor(buf45, (8, 384, 196), (75264, 196, 1), 0); del buf45  # reuse
    cpp_fused_add_clone_native_layer_norm_21(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg62_1
    del arg63_1
    buf79 = buf65; del buf65  # reuse
    # Source Nodes: [x_74], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf78, (3072, 196), (196, 1), 0), reinterpret_tensor(arg64_1, (196, 384), (1, 196), 0), out=buf79)
    del arg64_1
    buf80 = buf66; del buf66  # reuse
    cpp_fused_mul_silu_22(c_void_p(buf79.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(buf80.data_ptr()))
    del arg65_1
    buf81 = reinterpret_tensor(buf78, (3072, 196), (196, 1), 0); del buf78  # reuse
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg67_1, reinterpret_tensor(buf80, (3072, 192), (192, 1), 0), reinterpret_tensor(arg66_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf81)
    del arg66_1
    del arg67_1
    buf82 = buf76; del buf76  # reuse
    buf83 = buf75; del buf75  # reuse
    buf85 = reinterpret_tensor(buf38, (8, 196, 384), (75264, 384, 1), 0); del buf38  # reuse
    cpp_fused_add_native_layer_norm_23(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf85.data_ptr()))
    del arg68_1
    del arg69_1
    buf86 = buf72; del buf72  # reuse
    # Source Nodes: [x_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf85, (1568, 384), (384, 1), 0), reinterpret_tensor(arg70_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf86)
    del arg70_1
    del arg71_1
    buf87 = buf73; del buf73  # reuse
    cpp_fused_mul_silu_24(c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    buf88 = reinterpret_tensor(buf85, (1568, 384), (384, 1), 0); del buf85  # reuse
    # Source Nodes: [x_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf87, (1568, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf88)
    del arg72_1
    del arg73_1
    buf89 = reinterpret_tensor(buf88, (8, 196, 384), (75264, 384, 1), 0); del buf88  # reuse
    buf90 = buf83; del buf83  # reuse
    buf91 = buf82; del buf82  # reuse
    buf93 = reinterpret_tensor(buf31, (8, 384, 196), (75264, 196, 1), 0); del buf31  # reuse
    cpp_fused_add_clone_native_layer_norm_25(c_void_p(buf89.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()))
    del arg74_1
    del arg75_1
    buf94 = buf79; del buf79  # reuse
    # Source Nodes: [x_88], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf93, (3072, 196), (196, 1), 0), reinterpret_tensor(arg76_1, (196, 384), (1, 196), 0), out=buf94)
    del arg76_1
    buf95 = buf80; del buf80  # reuse
    cpp_fused_mul_silu_26(c_void_p(buf94.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf95.data_ptr()))
    del arg77_1
    buf96 = reinterpret_tensor(buf93, (3072, 196), (196, 1), 0); del buf93  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg79_1, reinterpret_tensor(buf95, (3072, 192), (192, 1), 0), reinterpret_tensor(arg78_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf96)
    del arg78_1
    del arg79_1
    buf97 = buf91; del buf91  # reuse
    buf98 = buf90; del buf90  # reuse
    buf100 = reinterpret_tensor(buf81, (8, 196, 384), (75264, 384, 1), 0); del buf81  # reuse
    cpp_fused_add_native_layer_norm_27(c_void_p(buf89.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf100.data_ptr()))
    del arg80_1
    del arg81_1
    buf101 = buf86; del buf86  # reuse
    # Source Nodes: [x_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg83_1, reinterpret_tensor(buf100, (1568, 384), (384, 1), 0), reinterpret_tensor(arg82_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf101)
    del arg82_1
    del arg83_1
    buf102 = buf87; del buf87  # reuse
    cpp_fused_mul_silu_28(c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    buf103 = reinterpret_tensor(buf100, (1568, 384), (384, 1), 0); del buf100  # reuse
    # Source Nodes: [x_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf102, (1568, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf103)
    del arg84_1
    del arg85_1
    buf104 = buf98; del buf98  # reuse
    buf105 = buf97; del buf97  # reuse
    buf107 = reinterpret_tensor(buf74, (8, 384, 196), (75264, 196, 1), 0); del buf74  # reuse
    cpp_fused_add_clone_native_layer_norm_29(c_void_p(buf89.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf107.data_ptr()))
    del arg86_1
    del arg87_1
    buf108 = buf94; del buf94  # reuse
    # Source Nodes: [x_102], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (3072, 196), (196, 1), 0), reinterpret_tensor(arg88_1, (196, 384), (1, 196), 0), out=buf108)
    del arg88_1
    buf109 = buf95; del buf95  # reuse
    cpp_fused_mul_silu_30(c_void_p(buf108.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg89_1
    buf110 = reinterpret_tensor(buf107, (3072, 196), (196, 1), 0); del buf107  # reuse
    # Source Nodes: [x_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf109, (3072, 192), (192, 1), 0), reinterpret_tensor(arg90_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf110)
    del arg90_1
    del arg91_1
    buf111 = buf105; del buf105  # reuse
    buf112 = buf104; del buf104  # reuse
    buf114 = reinterpret_tensor(buf67, (8, 196, 384), (75264, 384, 1), 0); del buf67  # reuse
    cpp_fused_add_native_layer_norm_31(c_void_p(buf89.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()))
    del arg92_1
    del arg93_1
    buf115 = buf101; del buf101  # reuse
    # Source Nodes: [x_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, reinterpret_tensor(buf114, (1568, 384), (384, 1), 0), reinterpret_tensor(arg94_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf115)
    del arg94_1
    del arg95_1
    buf116 = buf102; del buf102  # reuse
    cpp_fused_mul_silu_32(c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    buf117 = reinterpret_tensor(buf114, (1568, 384), (384, 1), 0); del buf114  # reuse
    # Source Nodes: [x_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf116, (1568, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf117)
    del arg96_1
    del arg97_1
    buf118 = reinterpret_tensor(buf117, (8, 196, 384), (75264, 384, 1), 0); del buf117  # reuse
    buf119 = buf112; del buf112  # reuse
    buf120 = buf111; del buf111  # reuse
    buf122 = reinterpret_tensor(buf60, (8, 384, 196), (75264, 196, 1), 0); del buf60  # reuse
    cpp_fused_add_clone_native_layer_norm_33(c_void_p(buf118.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()))
    del arg98_1
    del arg99_1
    buf123 = buf108; del buf108  # reuse
    # Source Nodes: [x_116], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf122, (3072, 196), (196, 1), 0), reinterpret_tensor(arg100_1, (196, 384), (1, 196), 0), out=buf123)
    del arg100_1
    buf124 = buf109; del buf109  # reuse
    cpp_fused_mul_silu_34(c_void_p(buf123.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(buf124.data_ptr()))
    del arg101_1
    buf125 = reinterpret_tensor(buf122, (3072, 196), (196, 1), 0); del buf122  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf124, (3072, 192), (192, 1), 0), reinterpret_tensor(arg102_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf125)
    del arg102_1
    del arg103_1
    buf126 = buf120; del buf120  # reuse
    buf127 = buf119; del buf119  # reuse
    buf129 = reinterpret_tensor(buf96, (8, 196, 384), (75264, 384, 1), 0); del buf96  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf118.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()))
    del arg104_1
    del arg105_1
    buf130 = buf115; del buf115  # reuse
    # Source Nodes: [x_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg107_1, reinterpret_tensor(buf129, (1568, 384), (384, 1), 0), reinterpret_tensor(arg106_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf130)
    del arg106_1
    del arg107_1
    buf131 = buf116; del buf116  # reuse
    cpp_fused_mul_silu_36(c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    buf132 = reinterpret_tensor(buf129, (1568, 384), (384, 1), 0); del buf129  # reuse
    # Source Nodes: [x_127], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg109_1, reinterpret_tensor(buf131, (1568, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf132)
    del arg108_1
    del arg109_1
    buf133 = buf127; del buf127  # reuse
    buf134 = buf126; del buf126  # reuse
    buf136 = reinterpret_tensor(buf89, (8, 384, 196), (75264, 196, 1), 0); del buf89  # reuse
    cpp_fused_add_clone_native_layer_norm_37(c_void_p(buf118.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()))
    del arg110_1
    del arg111_1
    buf137 = buf123; del buf123  # reuse
    # Source Nodes: [x_130], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (3072, 196), (196, 1), 0), reinterpret_tensor(arg112_1, (196, 384), (1, 196), 0), out=buf137)
    del arg112_1
    buf138 = buf124; del buf124  # reuse
    cpp_fused_mul_silu_38(c_void_p(buf137.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(buf138.data_ptr()))
    del arg113_1
    buf139 = reinterpret_tensor(buf136, (3072, 196), (196, 1), 0); del buf136  # reuse
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf138, (3072, 192), (192, 1), 0), reinterpret_tensor(arg114_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf139)
    del arg114_1
    del arg115_1
    buf140 = buf134; del buf134  # reuse
    buf141 = buf133; del buf133  # reuse
    buf143 = reinterpret_tensor(buf110, (8, 196, 384), (75264, 384, 1), 0); del buf110  # reuse
    cpp_fused_add_native_layer_norm_39(c_void_p(buf118.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()))
    del arg116_1
    del arg117_1
    buf144 = buf130; del buf130  # reuse
    # Source Nodes: [x_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf143, (1568, 384), (384, 1), 0), reinterpret_tensor(arg118_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf144)
    del arg118_1
    del arg119_1
    buf145 = buf131; del buf131  # reuse
    cpp_fused_mul_silu_40(c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    buf146 = reinterpret_tensor(buf143, (1568, 384), (384, 1), 0); del buf143  # reuse
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf145, (1568, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf146)
    del arg120_1
    del arg121_1
    buf147 = reinterpret_tensor(buf146, (8, 196, 384), (75264, 384, 1), 0); del buf146  # reuse
    buf148 = buf141; del buf141  # reuse
    buf149 = buf140; del buf140  # reuse
    buf151 = reinterpret_tensor(buf103, (8, 384, 196), (75264, 196, 1), 0); del buf103  # reuse
    cpp_fused_add_clone_native_layer_norm_41(c_void_p(buf147.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()))
    del arg122_1
    del arg123_1
    buf152 = buf137; del buf137  # reuse
    # Source Nodes: [x_144], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf151, (3072, 196), (196, 1), 0), reinterpret_tensor(arg124_1, (196, 384), (1, 196), 0), out=buf152)
    del arg124_1
    buf153 = buf138; del buf138  # reuse
    cpp_fused_mul_silu_42(c_void_p(buf152.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg125_1
    buf154 = reinterpret_tensor(buf151, (3072, 196), (196, 1), 0); del buf151  # reuse
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg127_1, reinterpret_tensor(buf153, (3072, 192), (192, 1), 0), reinterpret_tensor(arg126_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf154)
    del arg126_1
    del arg127_1
    buf155 = buf149; del buf149  # reuse
    buf156 = buf148; del buf148  # reuse
    buf158 = reinterpret_tensor(buf139, (8, 196, 384), (75264, 384, 1), 0); del buf139  # reuse
    cpp_fused_add_native_layer_norm_43(c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()))
    del arg128_1
    del arg129_1
    buf159 = buf144; del buf144  # reuse
    # Source Nodes: [x_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf158, (1568, 384), (384, 1), 0), reinterpret_tensor(arg130_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf159)
    del arg130_1
    del arg131_1
    buf160 = buf145; del buf145  # reuse
    cpp_fused_mul_silu_44(c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    buf161 = reinterpret_tensor(buf158, (1568, 384), (384, 1), 0); del buf158  # reuse
    # Source Nodes: [x_155], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg133_1, reinterpret_tensor(buf160, (1568, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf161)
    del arg132_1
    del arg133_1
    buf162 = buf156; del buf156  # reuse
    buf163 = buf155; del buf155  # reuse
    buf165 = reinterpret_tensor(buf132, (8, 384, 196), (75264, 196, 1), 0); del buf132  # reuse
    cpp_fused_add_clone_native_layer_norm_45(c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()))
    del arg134_1
    del arg135_1
    buf166 = buf152; del buf152  # reuse
    # Source Nodes: [x_158], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf165, (3072, 196), (196, 1), 0), reinterpret_tensor(arg136_1, (196, 384), (1, 196), 0), out=buf166)
    del arg136_1
    buf167 = buf153; del buf153  # reuse
    cpp_fused_mul_silu_46(c_void_p(buf166.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(buf167.data_ptr()))
    del arg137_1
    buf168 = reinterpret_tensor(buf165, (3072, 196), (196, 1), 0); del buf165  # reuse
    # Source Nodes: [x_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg139_1, reinterpret_tensor(buf167, (3072, 192), (192, 1), 0), reinterpret_tensor(arg138_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf168)
    del arg138_1
    del arg139_1
    buf169 = buf163; del buf163  # reuse
    buf170 = buf162; del buf162  # reuse
    buf172 = reinterpret_tensor(buf125, (8, 196, 384), (75264, 384, 1), 0); del buf125  # reuse
    cpp_fused_add_native_layer_norm_47(c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()))
    del arg140_1
    del arg141_1
    buf173 = buf159; del buf159  # reuse
    # Source Nodes: [x_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf172, (1568, 384), (384, 1), 0), reinterpret_tensor(arg142_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf173)
    del arg142_1
    del arg143_1
    buf174 = buf160; del buf160  # reuse
    cpp_fused_mul_silu_48(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    buf175 = reinterpret_tensor(buf172, (1568, 384), (384, 1), 0); del buf172  # reuse
    # Source Nodes: [x_169], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf174, (1568, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf175)
    del arg144_1
    del arg145_1
    buf176 = reinterpret_tensor(buf175, (8, 196, 384), (75264, 384, 1), 0); del buf175  # reuse
    buf177 = buf170; del buf170  # reuse
    buf178 = buf169; del buf169  # reuse
    buf180 = reinterpret_tensor(buf118, (8, 384, 196), (75264, 196, 1), 0); del buf118  # reuse
    cpp_fused_add_clone_native_layer_norm_49(c_void_p(buf176.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    del arg146_1
    del arg147_1
    buf181 = buf166; del buf166  # reuse
    # Source Nodes: [x_172], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf180, (3072, 196), (196, 1), 0), reinterpret_tensor(arg148_1, (196, 384), (1, 196), 0), out=buf181)
    del arg148_1
    buf182 = buf167; del buf167  # reuse
    cpp_fused_mul_silu_50(c_void_p(buf181.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(buf182.data_ptr()))
    del arg149_1
    buf183 = reinterpret_tensor(buf180, (3072, 196), (196, 1), 0); del buf180  # reuse
    # Source Nodes: [x_176], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg151_1, reinterpret_tensor(buf182, (3072, 192), (192, 1), 0), reinterpret_tensor(arg150_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf183)
    del arg150_1
    del arg151_1
    buf184 = buf178; del buf178  # reuse
    buf185 = buf177; del buf177  # reuse
    buf187 = reinterpret_tensor(buf168, (8, 196, 384), (75264, 384, 1), 0); del buf168  # reuse
    cpp_fused_add_native_layer_norm_51(c_void_p(buf176.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()))
    del arg152_1
    del arg153_1
    buf188 = buf173; del buf173  # reuse
    # Source Nodes: [x_179], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg155_1, reinterpret_tensor(buf187, (1568, 384), (384, 1), 0), reinterpret_tensor(arg154_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf188)
    del arg154_1
    del arg155_1
    buf189 = buf174; del buf174  # reuse
    cpp_fused_mul_silu_52(c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = reinterpret_tensor(buf187, (1568, 384), (384, 1), 0); del buf187  # reuse
    # Source Nodes: [x_183], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg157_1, reinterpret_tensor(buf189, (1568, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf190)
    del arg156_1
    del arg157_1
    buf191 = buf185; del buf185  # reuse
    buf192 = buf184; del buf184  # reuse
    buf194 = reinterpret_tensor(buf161, (8, 384, 196), (75264, 196, 1), 0); del buf161  # reuse
    cpp_fused_add_clone_native_layer_norm_53(c_void_p(buf176.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf194.data_ptr()))
    del arg158_1
    del arg159_1
    buf195 = buf181; del buf181  # reuse
    # Source Nodes: [x_186], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (3072, 196), (196, 1), 0), reinterpret_tensor(arg160_1, (196, 384), (1, 196), 0), out=buf195)
    del arg160_1
    buf196 = buf182; del buf182  # reuse
    cpp_fused_mul_silu_54(c_void_p(buf195.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(buf196.data_ptr()))
    del arg161_1
    buf197 = reinterpret_tensor(buf194, (3072, 196), (196, 1), 0); del buf194  # reuse
    # Source Nodes: [x_190], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg163_1, reinterpret_tensor(buf196, (3072, 192), (192, 1), 0), reinterpret_tensor(arg162_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf197)
    del arg162_1
    del arg163_1
    buf198 = buf192; del buf192  # reuse
    buf199 = buf191; del buf191  # reuse
    buf201 = reinterpret_tensor(buf154, (8, 196, 384), (75264, 384, 1), 0); del buf154  # reuse
    cpp_fused_add_native_layer_norm_55(c_void_p(buf176.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf201.data_ptr()))
    del arg164_1
    del arg165_1
    buf202 = buf188; del buf188  # reuse
    # Source Nodes: [x_193], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg167_1, reinterpret_tensor(buf201, (1568, 384), (384, 1), 0), reinterpret_tensor(arg166_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf202)
    del arg166_1
    del arg167_1
    buf203 = buf189; del buf189  # reuse
    cpp_fused_mul_silu_56(c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    buf204 = reinterpret_tensor(buf201, (1568, 384), (384, 1), 0); del buf201  # reuse
    # Source Nodes: [x_197], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg169_1, reinterpret_tensor(buf203, (1568, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf204)
    del arg168_1
    del arg169_1
    buf205 = reinterpret_tensor(buf204, (8, 196, 384), (75264, 384, 1), 0); del buf204  # reuse
    buf206 = buf199; del buf199  # reuse
    buf207 = buf198; del buf198  # reuse
    buf209 = reinterpret_tensor(buf147, (8, 384, 196), (75264, 196, 1), 0); del buf147  # reuse
    cpp_fused_add_clone_native_layer_norm_57(c_void_p(buf205.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf209.data_ptr()))
    del arg170_1
    del arg171_1
    buf210 = buf195; del buf195  # reuse
    # Source Nodes: [x_200], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (3072, 196), (196, 1), 0), reinterpret_tensor(arg172_1, (196, 384), (1, 196), 0), out=buf210)
    del arg172_1
    buf211 = buf196; del buf196  # reuse
    cpp_fused_mul_silu_58(c_void_p(buf210.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf211.data_ptr()))
    del arg173_1
    buf212 = reinterpret_tensor(buf209, (3072, 196), (196, 1), 0); del buf209  # reuse
    # Source Nodes: [x_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg175_1, reinterpret_tensor(buf211, (3072, 192), (192, 1), 0), reinterpret_tensor(arg174_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf212)
    del arg174_1
    del arg175_1
    buf213 = buf207; del buf207  # reuse
    buf214 = buf206; del buf206  # reuse
    buf216 = reinterpret_tensor(buf197, (8, 196, 384), (75264, 384, 1), 0); del buf197  # reuse
    cpp_fused_add_native_layer_norm_59(c_void_p(buf205.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()))
    del arg176_1
    del arg177_1
    buf217 = buf202; del buf202  # reuse
    # Source Nodes: [x_207], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg179_1, reinterpret_tensor(buf216, (1568, 384), (384, 1), 0), reinterpret_tensor(arg178_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf217)
    del arg178_1
    del arg179_1
    buf218 = buf203; del buf203  # reuse
    cpp_fused_mul_silu_60(c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    buf219 = reinterpret_tensor(buf216, (1568, 384), (384, 1), 0); del buf216  # reuse
    # Source Nodes: [x_211], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg181_1, reinterpret_tensor(buf218, (1568, 768), (768, 1), 0), reinterpret_tensor(arg180_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf219)
    del arg180_1
    del arg181_1
    buf220 = buf214; del buf214  # reuse
    buf221 = buf213; del buf213  # reuse
    buf223 = reinterpret_tensor(buf190, (8, 384, 196), (75264, 196, 1), 0); del buf190  # reuse
    cpp_fused_add_clone_native_layer_norm_61(c_void_p(buf205.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()))
    del arg182_1
    del arg183_1
    buf224 = buf210; del buf210  # reuse
    # Source Nodes: [x_214], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (3072, 196), (196, 1), 0), reinterpret_tensor(arg184_1, (196, 384), (1, 196), 0), out=buf224)
    del arg184_1
    buf225 = buf211; del buf211  # reuse
    cpp_fused_mul_silu_62(c_void_p(buf224.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(buf225.data_ptr()))
    del arg185_1
    buf226 = reinterpret_tensor(buf223, (3072, 196), (196, 1), 0); del buf223  # reuse
    # Source Nodes: [x_218], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg187_1, reinterpret_tensor(buf225, (3072, 192), (192, 1), 0), reinterpret_tensor(arg186_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf226)
    del arg186_1
    del arg187_1
    buf227 = buf221; del buf221  # reuse
    buf228 = buf220; del buf220  # reuse
    buf230 = reinterpret_tensor(buf183, (8, 196, 384), (75264, 384, 1), 0); del buf183  # reuse
    cpp_fused_add_native_layer_norm_63(c_void_p(buf205.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf230.data_ptr()))
    del arg188_1
    del arg189_1
    buf231 = buf217; del buf217  # reuse
    # Source Nodes: [x_221], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg191_1, reinterpret_tensor(buf230, (1568, 384), (384, 1), 0), reinterpret_tensor(arg190_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf231)
    del arg190_1
    del arg191_1
    buf232 = buf218; del buf218  # reuse
    cpp_fused_mul_silu_64(c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    buf233 = reinterpret_tensor(buf230, (1568, 384), (384, 1), 0); del buf230  # reuse
    # Source Nodes: [x_225], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg193_1, reinterpret_tensor(buf232, (1568, 768), (768, 1), 0), reinterpret_tensor(arg192_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf233)
    del arg192_1
    del arg193_1
    buf234 = reinterpret_tensor(buf233, (8, 196, 384), (75264, 384, 1), 0); del buf233  # reuse
    buf235 = buf228; del buf228  # reuse
    buf236 = buf227; del buf227  # reuse
    buf238 = reinterpret_tensor(buf176, (8, 384, 196), (75264, 196, 1), 0); del buf176  # reuse
    cpp_fused_add_clone_native_layer_norm_65(c_void_p(buf234.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()))
    del arg194_1
    del arg195_1
    buf239 = buf224; del buf224  # reuse
    # Source Nodes: [x_228], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf238, (3072, 196), (196, 1), 0), reinterpret_tensor(arg196_1, (196, 384), (1, 196), 0), out=buf239)
    del arg196_1
    buf240 = buf225; del buf225  # reuse
    cpp_fused_mul_silu_66(c_void_p(buf239.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(buf240.data_ptr()))
    del arg197_1
    buf241 = reinterpret_tensor(buf238, (3072, 196), (196, 1), 0); del buf238  # reuse
    # Source Nodes: [x_232], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg199_1, reinterpret_tensor(buf240, (3072, 192), (192, 1), 0), reinterpret_tensor(arg198_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf241)
    del arg198_1
    del arg199_1
    buf242 = buf236; del buf236  # reuse
    buf243 = buf235; del buf235  # reuse
    buf245 = reinterpret_tensor(buf226, (8, 196, 384), (75264, 384, 1), 0); del buf226  # reuse
    cpp_fused_add_native_layer_norm_67(c_void_p(buf234.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()))
    del arg200_1
    del arg201_1
    buf246 = buf231; del buf231  # reuse
    # Source Nodes: [x_235], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg203_1, reinterpret_tensor(buf245, (1568, 384), (384, 1), 0), reinterpret_tensor(arg202_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf246)
    del arg202_1
    del arg203_1
    buf247 = buf232; del buf232  # reuse
    cpp_fused_mul_silu_68(c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    buf248 = reinterpret_tensor(buf245, (1568, 384), (384, 1), 0); del buf245  # reuse
    # Source Nodes: [x_239], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg205_1, reinterpret_tensor(buf247, (1568, 768), (768, 1), 0), reinterpret_tensor(arg204_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf248)
    del arg204_1
    del arg205_1
    buf249 = buf243; del buf243  # reuse
    buf250 = buf242; del buf242  # reuse
    buf252 = reinterpret_tensor(buf219, (8, 384, 196), (75264, 196, 1), 0); del buf219  # reuse
    cpp_fused_add_clone_native_layer_norm_69(c_void_p(buf234.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf252.data_ptr()))
    del arg206_1
    del arg207_1
    buf253 = buf239; del buf239  # reuse
    # Source Nodes: [x_242], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf252, (3072, 196), (196, 1), 0), reinterpret_tensor(arg208_1, (196, 384), (1, 196), 0), out=buf253)
    del arg208_1
    buf254 = buf240; del buf240  # reuse
    cpp_fused_mul_silu_70(c_void_p(buf253.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(buf254.data_ptr()))
    del arg209_1
    buf255 = reinterpret_tensor(buf252, (3072, 196), (196, 1), 0); del buf252  # reuse
    # Source Nodes: [x_246], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg211_1, reinterpret_tensor(buf254, (3072, 192), (192, 1), 0), reinterpret_tensor(arg210_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf255)
    del arg210_1
    del arg211_1
    buf256 = buf250; del buf250  # reuse
    buf257 = buf249; del buf249  # reuse
    buf259 = reinterpret_tensor(buf212, (8, 196, 384), (75264, 384, 1), 0); del buf212  # reuse
    cpp_fused_add_native_layer_norm_71(c_void_p(buf234.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()))
    del arg212_1
    del arg213_1
    buf260 = buf246; del buf246  # reuse
    # Source Nodes: [x_249], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg215_1, reinterpret_tensor(buf259, (1568, 384), (384, 1), 0), reinterpret_tensor(arg214_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf260)
    del arg214_1
    del arg215_1
    buf261 = buf247; del buf247  # reuse
    cpp_fused_mul_silu_72(c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    buf262 = reinterpret_tensor(buf259, (1568, 384), (384, 1), 0); del buf259  # reuse
    # Source Nodes: [x_253], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg217_1, reinterpret_tensor(buf261, (1568, 768), (768, 1), 0), reinterpret_tensor(arg216_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf262)
    del arg216_1
    del arg217_1
    buf263 = reinterpret_tensor(buf262, (8, 196, 384), (75264, 384, 1), 0); del buf262  # reuse
    buf264 = buf257; del buf257  # reuse
    buf265 = buf256; del buf256  # reuse
    buf267 = reinterpret_tensor(buf205, (8, 384, 196), (75264, 196, 1), 0); del buf205  # reuse
    cpp_fused_add_clone_native_layer_norm_73(c_void_p(buf263.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf267.data_ptr()))
    del arg218_1
    del arg219_1
    buf268 = buf253; del buf253  # reuse
    # Source Nodes: [x_256], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf267, (3072, 196), (196, 1), 0), reinterpret_tensor(arg220_1, (196, 384), (1, 196), 0), out=buf268)
    del arg220_1
    buf269 = buf254; del buf254  # reuse
    cpp_fused_mul_silu_74(c_void_p(buf268.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(buf269.data_ptr()))
    del arg221_1
    buf270 = reinterpret_tensor(buf267, (3072, 196), (196, 1), 0); del buf267  # reuse
    # Source Nodes: [x_260], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg223_1, reinterpret_tensor(buf269, (3072, 192), (192, 1), 0), reinterpret_tensor(arg222_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf270)
    del arg222_1
    del arg223_1
    buf271 = buf265; del buf265  # reuse
    buf272 = buf264; del buf264  # reuse
    buf274 = reinterpret_tensor(buf255, (8, 196, 384), (75264, 384, 1), 0); del buf255  # reuse
    cpp_fused_add_native_layer_norm_75(c_void_p(buf263.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf274.data_ptr()))
    del arg224_1
    del arg225_1
    buf275 = buf260; del buf260  # reuse
    # Source Nodes: [x_263], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg227_1, reinterpret_tensor(buf274, (1568, 384), (384, 1), 0), reinterpret_tensor(arg226_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf275)
    del arg226_1
    del arg227_1
    buf276 = buf261; del buf261  # reuse
    cpp_fused_mul_silu_76(c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    buf277 = reinterpret_tensor(buf274, (1568, 384), (384, 1), 0); del buf274  # reuse
    # Source Nodes: [x_267], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg229_1, reinterpret_tensor(buf276, (1568, 768), (768, 1), 0), reinterpret_tensor(arg228_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf277)
    del arg228_1
    del arg229_1
    buf278 = buf272; del buf272  # reuse
    buf279 = buf271; del buf271  # reuse
    buf281 = reinterpret_tensor(buf248, (8, 384, 196), (75264, 196, 1), 0); del buf248  # reuse
    cpp_fused_add_clone_native_layer_norm_77(c_void_p(buf263.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()))
    del arg230_1
    del arg231_1
    buf282 = buf268; del buf268  # reuse
    # Source Nodes: [x_270], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf281, (3072, 196), (196, 1), 0), reinterpret_tensor(arg232_1, (196, 384), (1, 196), 0), out=buf282)
    del arg232_1
    buf283 = buf269; del buf269  # reuse
    cpp_fused_mul_silu_78(c_void_p(buf282.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(buf283.data_ptr()))
    del arg233_1
    buf284 = reinterpret_tensor(buf281, (3072, 196), (196, 1), 0); del buf281  # reuse
    # Source Nodes: [x_274], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg235_1, reinterpret_tensor(buf283, (3072, 192), (192, 1), 0), reinterpret_tensor(arg234_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf284)
    del arg234_1
    del arg235_1
    buf285 = buf279; del buf279  # reuse
    buf286 = buf278; del buf278  # reuse
    buf288 = reinterpret_tensor(buf241, (8, 196, 384), (75264, 384, 1), 0); del buf241  # reuse
    cpp_fused_add_native_layer_norm_79(c_void_p(buf263.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()))
    del arg236_1
    del arg237_1
    buf289 = buf275; del buf275  # reuse
    # Source Nodes: [x_277], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg239_1, reinterpret_tensor(buf288, (1568, 384), (384, 1), 0), reinterpret_tensor(arg238_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf289)
    del arg238_1
    del arg239_1
    buf290 = buf276; del buf276  # reuse
    cpp_fused_mul_silu_80(c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()))
    buf291 = reinterpret_tensor(buf288, (1568, 384), (384, 1), 0); del buf288  # reuse
    # Source Nodes: [x_281], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg241_1, reinterpret_tensor(buf290, (1568, 768), (768, 1), 0), reinterpret_tensor(arg240_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf291)
    del arg240_1
    del arg241_1
    buf292 = reinterpret_tensor(buf291, (8, 196, 384), (75264, 384, 1), 0); del buf291  # reuse
    buf293 = buf286; del buf286  # reuse
    buf294 = buf285; del buf285  # reuse
    buf296 = reinterpret_tensor(buf234, (8, 384, 196), (75264, 196, 1), 0); del buf234  # reuse
    cpp_fused_add_clone_native_layer_norm_81(c_void_p(buf292.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()))
    del arg242_1
    del arg243_1
    buf297 = buf282; del buf282  # reuse
    # Source Nodes: [x_284], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (3072, 196), (196, 1), 0), reinterpret_tensor(arg244_1, (196, 384), (1, 196), 0), out=buf297)
    del arg244_1
    buf298 = buf283; del buf283  # reuse
    cpp_fused_mul_silu_82(c_void_p(buf297.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(buf298.data_ptr()))
    del arg245_1
    buf299 = reinterpret_tensor(buf296, (3072, 196), (196, 1), 0); del buf296  # reuse
    # Source Nodes: [x_288], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg247_1, reinterpret_tensor(buf298, (3072, 192), (192, 1), 0), reinterpret_tensor(arg246_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf299)
    del arg246_1
    del arg247_1
    buf300 = buf294; del buf294  # reuse
    buf301 = buf293; del buf293  # reuse
    buf303 = reinterpret_tensor(buf284, (8, 196, 384), (75264, 384, 1), 0); del buf284  # reuse
    cpp_fused_add_native_layer_norm_83(c_void_p(buf292.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf303.data_ptr()))
    del arg248_1
    del arg249_1
    buf304 = buf289; del buf289  # reuse
    # Source Nodes: [x_291], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg251_1, reinterpret_tensor(buf303, (1568, 384), (384, 1), 0), reinterpret_tensor(arg250_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf304)
    del arg250_1
    del arg251_1
    buf305 = buf290; del buf290  # reuse
    cpp_fused_mul_silu_84(c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    buf306 = reinterpret_tensor(buf303, (1568, 384), (384, 1), 0); del buf303  # reuse
    # Source Nodes: [x_295], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg253_1, reinterpret_tensor(buf305, (1568, 768), (768, 1), 0), reinterpret_tensor(arg252_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf306)
    del arg252_1
    del arg253_1
    buf307 = buf301; del buf301  # reuse
    buf308 = buf300; del buf300  # reuse
    buf310 = reinterpret_tensor(buf277, (8, 384, 196), (75264, 196, 1), 0); del buf277  # reuse
    cpp_fused_add_clone_native_layer_norm_85(c_void_p(buf292.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf310.data_ptr()))
    del arg254_1
    del arg255_1
    buf311 = buf297; del buf297  # reuse
    # Source Nodes: [x_298], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf310, (3072, 196), (196, 1), 0), reinterpret_tensor(arg256_1, (196, 384), (1, 196), 0), out=buf311)
    del arg256_1
    buf312 = buf298; del buf298  # reuse
    cpp_fused_mul_silu_86(c_void_p(buf311.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(buf312.data_ptr()))
    del arg257_1
    buf313 = reinterpret_tensor(buf310, (3072, 196), (196, 1), 0); del buf310  # reuse
    # Source Nodes: [x_302], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg259_1, reinterpret_tensor(buf312, (3072, 192), (192, 1), 0), reinterpret_tensor(arg258_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf313)
    del arg258_1
    del arg259_1
    buf314 = buf308; del buf308  # reuse
    buf315 = buf307; del buf307  # reuse
    buf317 = reinterpret_tensor(buf270, (8, 196, 384), (75264, 384, 1), 0); del buf270  # reuse
    cpp_fused_add_native_layer_norm_87(c_void_p(buf292.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf317.data_ptr()))
    del arg260_1
    del arg261_1
    buf318 = buf304; del buf304  # reuse
    # Source Nodes: [x_305], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg263_1, reinterpret_tensor(buf317, (1568, 384), (384, 1), 0), reinterpret_tensor(arg262_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf318)
    del arg262_1
    del arg263_1
    buf319 = buf305; del buf305  # reuse
    cpp_fused_mul_silu_88(c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    buf320 = reinterpret_tensor(buf317, (1568, 384), (384, 1), 0); del buf317  # reuse
    # Source Nodes: [x_309], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg265_1, reinterpret_tensor(buf319, (1568, 768), (768, 1), 0), reinterpret_tensor(arg264_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf320)
    del arg264_1
    del arg265_1
    buf321 = reinterpret_tensor(buf320, (8, 196, 384), (75264, 384, 1), 0); del buf320  # reuse
    buf322 = buf315; del buf315  # reuse
    buf323 = buf314; del buf314  # reuse
    buf325 = reinterpret_tensor(buf263, (8, 384, 196), (75264, 196, 1), 0); del buf263  # reuse
    cpp_fused_add_clone_native_layer_norm_89(c_void_p(buf321.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf325.data_ptr()))
    del arg266_1
    del arg267_1
    del buf292
    buf326 = buf311; del buf311  # reuse
    # Source Nodes: [x_312], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf325, (3072, 196), (196, 1), 0), reinterpret_tensor(arg268_1, (196, 384), (1, 196), 0), out=buf326)
    del arg268_1
    buf327 = buf312; del buf312  # reuse
    cpp_fused_mul_silu_90(c_void_p(buf326.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(buf327.data_ptr()))
    del arg269_1
    buf328 = reinterpret_tensor(buf325, (3072, 196), (196, 1), 0); del buf325  # reuse
    # Source Nodes: [x_316], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg271_1, reinterpret_tensor(buf327, (3072, 192), (192, 1), 0), reinterpret_tensor(arg270_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf328)
    del arg270_1
    del arg271_1
    buf329 = buf323; del buf323  # reuse
    buf330 = buf322; del buf322  # reuse
    buf332 = reinterpret_tensor(buf313, (8, 196, 384), (75264, 384, 1), 0); del buf313  # reuse
    cpp_fused_add_native_layer_norm_91(c_void_p(buf321.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf332.data_ptr()))
    del arg272_1
    del arg273_1
    buf333 = buf318; del buf318  # reuse
    # Source Nodes: [x_319], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg275_1, reinterpret_tensor(buf332, (1568, 384), (384, 1), 0), reinterpret_tensor(arg274_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf333)
    del arg274_1
    del arg275_1
    buf334 = buf319; del buf319  # reuse
    cpp_fused_mul_silu_92(c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()))
    buf335 = reinterpret_tensor(buf332, (1568, 384), (384, 1), 0); del buf332  # reuse
    # Source Nodes: [x_323], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg277_1, reinterpret_tensor(buf334, (1568, 768), (768, 1), 0), reinterpret_tensor(arg276_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf335)
    del arg276_1
    del arg277_1
    buf336 = buf330; del buf330  # reuse
    buf337 = buf329; del buf329  # reuse
    buf339 = reinterpret_tensor(buf306, (8, 384, 196), (75264, 196, 1), 0); del buf306  # reuse
    cpp_fused_add_clone_native_layer_norm_93(c_void_p(buf321.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf339.data_ptr()))
    del arg278_1
    del arg279_1
    buf340 = buf326; del buf326  # reuse
    # Source Nodes: [x_326], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf339, (3072, 196), (196, 1), 0), reinterpret_tensor(arg280_1, (196, 384), (1, 196), 0), out=buf340)
    del arg280_1
    buf341 = buf327; del buf327  # reuse
    cpp_fused_mul_silu_94(c_void_p(buf340.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(buf341.data_ptr()))
    del arg281_1
    del buf340
    buf342 = reinterpret_tensor(buf339, (3072, 196), (196, 1), 0); del buf339  # reuse
    # Source Nodes: [x_330], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg283_1, reinterpret_tensor(buf341, (3072, 192), (192, 1), 0), reinterpret_tensor(arg282_1, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf342)
    del arg282_1
    del arg283_1
    del buf341
    buf343 = buf337; del buf337  # reuse
    buf344 = buf336; del buf336  # reuse
    buf346 = reinterpret_tensor(buf299, (8, 196, 384), (75264, 384, 1), 0); del buf299  # reuse
    cpp_fused_add_native_layer_norm_95(c_void_p(buf321.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf346.data_ptr()))
    del arg284_1
    del arg285_1
    buf347 = buf333; del buf333  # reuse
    # Source Nodes: [x_333], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg287_1, reinterpret_tensor(buf346, (1568, 384), (384, 1), 0), reinterpret_tensor(arg286_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf347)
    del arg286_1
    del arg287_1
    buf348 = buf334; del buf334  # reuse
    cpp_fused_mul_silu_96(c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    del buf347
    buf349 = reinterpret_tensor(buf346, (1568, 384), (384, 1), 0); del buf346  # reuse
    # Source Nodes: [x_337], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg289_1, reinterpret_tensor(buf348, (1568, 768), (768, 1), 0), reinterpret_tensor(arg288_1, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf349)
    del arg288_1
    del arg289_1
    del buf348
    buf350 = reinterpret_tensor(buf349, (8, 196, 384), (75264, 384, 1), 0); del buf349  # reuse
    buf351 = buf344; del buf344  # reuse
    buf352 = buf343; del buf343  # reuse
    buf354 = empty((8, 384), device='cpu', dtype=torch.float32)
    buf355 = buf354; del buf354  # reuse
    cpp_fused_add_mean_native_layer_norm_97(c_void_p(buf350.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()))
    del arg290_1
    del arg291_1
    del buf321
    del buf328
    del buf335
    del buf342
    del buf350
    del buf351
    del buf352
    buf356 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_342, x_343, x_345], Original ATen: [aten.addmm, aten.mean, aten.native_layer_norm]
    extern_kernels.addmm(arg293_1, buf355, reinterpret_tensor(arg292_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf356)
    del arg292_1
    del arg293_1
    return (buf356, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmixer_24_224', benchmark_compiled_module)
