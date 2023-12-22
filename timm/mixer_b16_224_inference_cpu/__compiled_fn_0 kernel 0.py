
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(768.0);
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
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(768.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
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
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(768.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                            auto tmp13 = static_cast<float>(768.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(768.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp8;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(768.0);
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
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(768.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
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
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(768.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                            auto tmp13 = static_cast<float>(768.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(768.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp8;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(768.0);
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
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(768.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
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
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(768.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                            auto tmp13 = static_cast<float>(768.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(768.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp8;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(768.0);
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
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(768.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
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
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(768.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                            auto tmp13 = static_cast<float>(768.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(768.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp8;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(768.0);
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
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(768.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
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
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(768.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                            auto tmp13 = static_cast<float>(768.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(768.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp8;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp13 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(768.0);
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
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(768.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr2[static_cast<long>(x2)];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
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
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(768.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp2 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                            auto tmp13 = static_cast<float>(768.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = static_cast<float>(1e-06);
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = 1 / std::sqrt(tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            auto tmp23 = tmp21 + tmp22;
                            tmp23.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(768.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_native_layer_norm_49 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp1, 8);
                        float tmp6[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp6, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(tmp6 + static_cast<long>(8L*x1_inner));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp8 = tmp5 + tmp7;
                            auto tmp10 = tmp8 + tmp9;
                            tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp8;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                            auto tmp4 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(768.0);
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
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1 = args
    args.clear()
    assert_size_stride(arg0_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (384, 196), (196, 1))
    assert_size_stride(arg5_1, (384, ), (1, ))
    assert_size_stride(arg6_1, (196, 384), (384, 1))
    assert_size_stride(arg7_1, (196, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (3072, 768), (768, 1))
    assert_size_stride(arg11_1, (3072, ), (1, ))
    assert_size_stride(arg12_1, (768, 3072), (3072, 1))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (384, 196), (196, 1))
    assert_size_stride(arg17_1, (384, ), (1, ))
    assert_size_stride(arg18_1, (196, 384), (384, 1))
    assert_size_stride(arg19_1, (196, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (3072, 768), (768, 1))
    assert_size_stride(arg23_1, (3072, ), (1, ))
    assert_size_stride(arg24_1, (768, 3072), (3072, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (384, 196), (196, 1))
    assert_size_stride(arg29_1, (384, ), (1, ))
    assert_size_stride(arg30_1, (196, 384), (384, 1))
    assert_size_stride(arg31_1, (196, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (3072, 768), (768, 1))
    assert_size_stride(arg35_1, (3072, ), (1, ))
    assert_size_stride(arg36_1, (768, 3072), (3072, 1))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (384, 196), (196, 1))
    assert_size_stride(arg41_1, (384, ), (1, ))
    assert_size_stride(arg42_1, (196, 384), (384, 1))
    assert_size_stride(arg43_1, (196, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (3072, 768), (768, 1))
    assert_size_stride(arg47_1, (3072, ), (1, ))
    assert_size_stride(arg48_1, (768, 3072), (3072, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (384, 196), (196, 1))
    assert_size_stride(arg53_1, (384, ), (1, ))
    assert_size_stride(arg54_1, (196, 384), (384, 1))
    assert_size_stride(arg55_1, (196, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (3072, 768), (768, 1))
    assert_size_stride(arg59_1, (3072, ), (1, ))
    assert_size_stride(arg60_1, (768, 3072), (3072, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (384, 196), (196, 1))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (196, 384), (384, 1))
    assert_size_stride(arg67_1, (196, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (3072, 768), (768, 1))
    assert_size_stride(arg71_1, (3072, ), (1, ))
    assert_size_stride(arg72_1, (768, 3072), (3072, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (384, 196), (196, 1))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (196, 384), (384, 1))
    assert_size_stride(arg79_1, (196, ), (1, ))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (3072, 768), (768, 1))
    assert_size_stride(arg83_1, (3072, ), (1, ))
    assert_size_stride(arg84_1, (768, 3072), (3072, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (384, 196), (196, 1))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (196, 384), (384, 1))
    assert_size_stride(arg91_1, (196, ), (1, ))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (3072, 768), (768, 1))
    assert_size_stride(arg95_1, (3072, ), (1, ))
    assert_size_stride(arg96_1, (768, 3072), (3072, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (384, 196), (196, 1))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (196, 384), (384, 1))
    assert_size_stride(arg103_1, (196, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (3072, 768), (768, 1))
    assert_size_stride(arg107_1, (3072, ), (1, ))
    assert_size_stride(arg108_1, (768, 3072), (3072, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (384, 196), (196, 1))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (196, 384), (384, 1))
    assert_size_stride(arg115_1, (196, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (3072, 768), (768, 1))
    assert_size_stride(arg119_1, (3072, ), (1, ))
    assert_size_stride(arg120_1, (768, 3072), (3072, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (384, 196), (196, 1))
    assert_size_stride(arg125_1, (384, ), (1, ))
    assert_size_stride(arg126_1, (196, 384), (384, 1))
    assert_size_stride(arg127_1, (196, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (3072, 768), (768, 1))
    assert_size_stride(arg131_1, (3072, ), (1, ))
    assert_size_stride(arg132_1, (768, 3072), (3072, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (384, 196), (196, 1))
    assert_size_stride(arg137_1, (384, ), (1, ))
    assert_size_stride(arg138_1, (196, 384), (384, 1))
    assert_size_stride(arg139_1, (196, ), (1, ))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (3072, 768), (768, 1))
    assert_size_stride(arg143_1, (3072, ), (1, ))
    assert_size_stride(arg144_1, (768, 3072), (3072, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (1000, 768), (768, 1))
    assert_size_stride(arg149_1, (1000, ), (1, ))
    assert_size_stride(arg150_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((768, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg150_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg150_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 768, 14, 14), (150528, 1, 10752, 768))
    del arg1_1
    del buf1
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf6 = reinterpret_tensor(buf0, (8, 768, 196), (150528, 196, 1), 0); del buf0  # reuse
    cpp_fused_clone_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg2_1
    del arg3_1
    buf7 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_4], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf6, (6144, 196), (196, 1), 0), reinterpret_tensor(arg4_1, (196, 384), (1, 196), 0), out=buf7)
    del arg4_1
    buf8 = reinterpret_tensor(buf7, (8, 768, 384), (294912, 384, 1), 0); del buf7  # reuse
    cpp_fused_add_gelu_2(c_void_p(buf8.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg5_1
    buf9 = reinterpret_tensor(buf6, (6144, 196), (196, 1), 0); del buf6  # reuse
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg7_1, reinterpret_tensor(buf8, (6144, 384), (384, 1), 0), reinterpret_tensor(arg6_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf9)
    del arg6_1
    del arg7_1
    buf10 = buf4; del buf4  # reuse
    buf11 = buf3; del buf3  # reuse
    buf13 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_3(c_void_p(buf2.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg8_1
    del arg9_1
    buf14 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg11_1, reinterpret_tensor(buf13, (1568, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf14)
    del arg10_1
    del arg11_1
    buf15 = reinterpret_tensor(buf14, (8, 196, 3072), (602112, 3072, 1), 0); del buf14  # reuse
    cpp_fused_gelu_4(c_void_p(buf15.data_ptr()))
    buf16 = reinterpret_tensor(buf13, (1568, 768), (768, 1), 0); del buf13  # reuse
    # Source Nodes: [x_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg13_1, reinterpret_tensor(buf15, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg12_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf16)
    del arg12_1
    del arg13_1
    buf17 = buf11; del buf11  # reuse
    buf18 = buf10; del buf10  # reuse
    buf20 = empty((8, 768, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_native_layer_norm_5(c_void_p(buf2.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()))
    del arg14_1
    del arg15_1
    buf21 = reinterpret_tensor(buf8, (6144, 384), (384, 1), 0); del buf8  # reuse
    # Source Nodes: [x_18], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (6144, 196), (196, 1), 0), reinterpret_tensor(arg16_1, (196, 384), (1, 196), 0), out=buf21)
    del arg16_1
    buf22 = reinterpret_tensor(buf21, (8, 768, 384), (294912, 384, 1), 0); del buf21  # reuse
    cpp_fused_add_gelu_6(c_void_p(buf22.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg17_1
    buf23 = reinterpret_tensor(buf20, (6144, 196), (196, 1), 0); del buf20  # reuse
    # Source Nodes: [x_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg19_1, reinterpret_tensor(buf22, (6144, 384), (384, 1), 0), reinterpret_tensor(arg18_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf23)
    del arg18_1
    del arg19_1
    buf24 = buf18; del buf18  # reuse
    buf25 = buf17; del buf17  # reuse
    buf27 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_7(c_void_p(buf2.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()))
    del arg20_1
    del arg21_1
    buf28 = reinterpret_tensor(buf15, (1568, 3072), (3072, 1), 0); del buf15  # reuse
    # Source Nodes: [x_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg23_1, reinterpret_tensor(buf27, (1568, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf28)
    del arg22_1
    del arg23_1
    buf29 = reinterpret_tensor(buf28, (8, 196, 3072), (602112, 3072, 1), 0); del buf28  # reuse
    cpp_fused_gelu_8(c_void_p(buf29.data_ptr()))
    buf30 = reinterpret_tensor(buf27, (1568, 768), (768, 1), 0); del buf27  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg25_1, reinterpret_tensor(buf29, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg24_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf30)
    del arg24_1
    del arg25_1
    buf31 = reinterpret_tensor(buf30, (8, 196, 768), (150528, 768, 1), 0); del buf30  # reuse
    buf32 = buf25; del buf25  # reuse
    buf33 = buf24; del buf24  # reuse
    buf35 = empty((8, 768, 196), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_native_layer_norm_9(c_void_p(buf31.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf35.data_ptr()))
    del arg26_1
    del arg27_1
    buf36 = reinterpret_tensor(buf22, (6144, 384), (384, 1), 0); del buf22  # reuse
    # Source Nodes: [x_32], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (6144, 196), (196, 1), 0), reinterpret_tensor(arg28_1, (196, 384), (1, 196), 0), out=buf36)
    del arg28_1
    buf37 = reinterpret_tensor(buf36, (8, 768, 384), (294912, 384, 1), 0); del buf36  # reuse
    cpp_fused_add_gelu_10(c_void_p(buf37.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg29_1
    buf38 = reinterpret_tensor(buf35, (6144, 196), (196, 1), 0); del buf35  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg31_1, reinterpret_tensor(buf37, (6144, 384), (384, 1), 0), reinterpret_tensor(arg30_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf38)
    del arg30_1
    del arg31_1
    buf39 = buf33; del buf33  # reuse
    buf40 = buf32; del buf32  # reuse
    buf42 = reinterpret_tensor(buf9, (8, 196, 768), (150528, 768, 1), 0); del buf9  # reuse
    cpp_fused_add_native_layer_norm_11(c_void_p(buf31.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()))
    del arg32_1
    del arg33_1
    buf43 = reinterpret_tensor(buf29, (1568, 3072), (3072, 1), 0); del buf29  # reuse
    # Source Nodes: [x_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg35_1, reinterpret_tensor(buf42, (1568, 768), (768, 1), 0), reinterpret_tensor(arg34_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf43)
    del arg34_1
    del arg35_1
    buf44 = reinterpret_tensor(buf43, (8, 196, 3072), (602112, 3072, 1), 0); del buf43  # reuse
    cpp_fused_gelu_12(c_void_p(buf44.data_ptr()))
    buf45 = reinterpret_tensor(buf42, (1568, 768), (768, 1), 0); del buf42  # reuse
    # Source Nodes: [x_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg37_1, reinterpret_tensor(buf44, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg36_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf45)
    del arg36_1
    del arg37_1
    buf46 = buf40; del buf40  # reuse
    buf47 = buf39; del buf39  # reuse
    buf49 = reinterpret_tensor(buf23, (8, 768, 196), (150528, 196, 1), 0); del buf23  # reuse
    cpp_fused_add_clone_native_layer_norm_13(c_void_p(buf31.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()))
    del arg38_1
    del arg39_1
    buf50 = reinterpret_tensor(buf37, (6144, 384), (384, 1), 0); del buf37  # reuse
    # Source Nodes: [x_46], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (6144, 196), (196, 1), 0), reinterpret_tensor(arg40_1, (196, 384), (1, 196), 0), out=buf50)
    del arg40_1
    buf51 = reinterpret_tensor(buf50, (8, 768, 384), (294912, 384, 1), 0); del buf50  # reuse
    cpp_fused_add_gelu_14(c_void_p(buf51.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg41_1
    buf52 = reinterpret_tensor(buf49, (6144, 196), (196, 1), 0); del buf49  # reuse
    # Source Nodes: [x_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg43_1, reinterpret_tensor(buf51, (6144, 384), (384, 1), 0), reinterpret_tensor(arg42_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf52)
    del arg42_1
    del arg43_1
    buf53 = buf47; del buf47  # reuse
    buf54 = buf46; del buf46  # reuse
    buf56 = reinterpret_tensor(buf2, (8, 196, 768), (150528, 768, 1), 0); del buf2  # reuse
    cpp_fused_add_native_layer_norm_15(c_void_p(buf31.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()))
    del arg44_1
    del arg45_1
    buf57 = reinterpret_tensor(buf44, (1568, 3072), (3072, 1), 0); del buf44  # reuse
    # Source Nodes: [x_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg47_1, reinterpret_tensor(buf56, (1568, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf57)
    del arg46_1
    del arg47_1
    buf58 = reinterpret_tensor(buf57, (8, 196, 3072), (602112, 3072, 1), 0); del buf57  # reuse
    cpp_fused_gelu_16(c_void_p(buf58.data_ptr()))
    buf59 = reinterpret_tensor(buf56, (1568, 768), (768, 1), 0); del buf56  # reuse
    # Source Nodes: [x_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg49_1, reinterpret_tensor(buf58, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg48_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf59)
    del arg48_1
    del arg49_1
    buf60 = reinterpret_tensor(buf59, (8, 196, 768), (150528, 768, 1), 0); del buf59  # reuse
    buf61 = buf54; del buf54  # reuse
    buf62 = buf53; del buf53  # reuse
    buf64 = reinterpret_tensor(buf16, (8, 768, 196), (150528, 196, 1), 0); del buf16  # reuse
    cpp_fused_add_clone_native_layer_norm_17(c_void_p(buf60.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()))
    del arg50_1
    del arg51_1
    buf65 = reinterpret_tensor(buf51, (6144, 384), (384, 1), 0); del buf51  # reuse
    # Source Nodes: [x_60], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (6144, 196), (196, 1), 0), reinterpret_tensor(arg52_1, (196, 384), (1, 196), 0), out=buf65)
    del arg52_1
    buf66 = reinterpret_tensor(buf65, (8, 768, 384), (294912, 384, 1), 0); del buf65  # reuse
    cpp_fused_add_gelu_18(c_void_p(buf66.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg53_1
    buf67 = reinterpret_tensor(buf64, (6144, 196), (196, 1), 0); del buf64  # reuse
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf66, (6144, 384), (384, 1), 0), reinterpret_tensor(arg54_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf67)
    del arg54_1
    del arg55_1
    buf68 = buf62; del buf62  # reuse
    buf69 = buf61; del buf61  # reuse
    buf71 = reinterpret_tensor(buf52, (8, 196, 768), (150528, 768, 1), 0); del buf52  # reuse
    cpp_fused_add_native_layer_norm_19(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()))
    del arg56_1
    del arg57_1
    buf72 = reinterpret_tensor(buf58, (1568, 3072), (3072, 1), 0); del buf58  # reuse
    # Source Nodes: [x_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg59_1, reinterpret_tensor(buf71, (1568, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf72)
    del arg58_1
    del arg59_1
    buf73 = reinterpret_tensor(buf72, (8, 196, 3072), (602112, 3072, 1), 0); del buf72  # reuse
    cpp_fused_gelu_20(c_void_p(buf73.data_ptr()))
    buf74 = reinterpret_tensor(buf71, (1568, 768), (768, 1), 0); del buf71  # reuse
    # Source Nodes: [x_71], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg61_1, reinterpret_tensor(buf73, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg60_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf74)
    del arg60_1
    del arg61_1
    buf75 = buf69; del buf69  # reuse
    buf76 = buf68; del buf68  # reuse
    buf78 = reinterpret_tensor(buf45, (8, 768, 196), (150528, 196, 1), 0); del buf45  # reuse
    cpp_fused_add_clone_native_layer_norm_21(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg62_1
    del arg63_1
    buf79 = reinterpret_tensor(buf66, (6144, 384), (384, 1), 0); del buf66  # reuse
    # Source Nodes: [x_74], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf78, (6144, 196), (196, 1), 0), reinterpret_tensor(arg64_1, (196, 384), (1, 196), 0), out=buf79)
    del arg64_1
    buf80 = reinterpret_tensor(buf79, (8, 768, 384), (294912, 384, 1), 0); del buf79  # reuse
    cpp_fused_add_gelu_22(c_void_p(buf80.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg65_1
    buf81 = reinterpret_tensor(buf78, (6144, 196), (196, 1), 0); del buf78  # reuse
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg67_1, reinterpret_tensor(buf80, (6144, 384), (384, 1), 0), reinterpret_tensor(arg66_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf81)
    del arg66_1
    del arg67_1
    buf82 = buf76; del buf76  # reuse
    buf83 = buf75; del buf75  # reuse
    buf85 = reinterpret_tensor(buf38, (8, 196, 768), (150528, 768, 1), 0); del buf38  # reuse
    cpp_fused_add_native_layer_norm_23(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf85.data_ptr()))
    del arg68_1
    del arg69_1
    buf86 = reinterpret_tensor(buf73, (1568, 3072), (3072, 1), 0); del buf73  # reuse
    # Source Nodes: [x_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf85, (1568, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf86)
    del arg70_1
    del arg71_1
    buf87 = reinterpret_tensor(buf86, (8, 196, 3072), (602112, 3072, 1), 0); del buf86  # reuse
    cpp_fused_gelu_24(c_void_p(buf87.data_ptr()))
    buf88 = reinterpret_tensor(buf85, (1568, 768), (768, 1), 0); del buf85  # reuse
    # Source Nodes: [x_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf87, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg72_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf88)
    del arg72_1
    del arg73_1
    buf89 = reinterpret_tensor(buf88, (8, 196, 768), (150528, 768, 1), 0); del buf88  # reuse
    buf90 = buf83; del buf83  # reuse
    buf91 = buf82; del buf82  # reuse
    buf93 = reinterpret_tensor(buf31, (8, 768, 196), (150528, 196, 1), 0); del buf31  # reuse
    cpp_fused_add_clone_native_layer_norm_25(c_void_p(buf89.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()))
    del arg74_1
    del arg75_1
    buf94 = reinterpret_tensor(buf80, (6144, 384), (384, 1), 0); del buf80  # reuse
    # Source Nodes: [x_88], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf93, (6144, 196), (196, 1), 0), reinterpret_tensor(arg76_1, (196, 384), (1, 196), 0), out=buf94)
    del arg76_1
    buf95 = reinterpret_tensor(buf94, (8, 768, 384), (294912, 384, 1), 0); del buf94  # reuse
    cpp_fused_add_gelu_26(c_void_p(buf95.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg77_1
    buf96 = reinterpret_tensor(buf93, (6144, 196), (196, 1), 0); del buf93  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg79_1, reinterpret_tensor(buf95, (6144, 384), (384, 1), 0), reinterpret_tensor(arg78_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf96)
    del arg78_1
    del arg79_1
    buf97 = buf91; del buf91  # reuse
    buf98 = buf90; del buf90  # reuse
    buf100 = reinterpret_tensor(buf81, (8, 196, 768), (150528, 768, 1), 0); del buf81  # reuse
    cpp_fused_add_native_layer_norm_27(c_void_p(buf89.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf100.data_ptr()))
    del arg80_1
    del arg81_1
    buf101 = reinterpret_tensor(buf87, (1568, 3072), (3072, 1), 0); del buf87  # reuse
    # Source Nodes: [x_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg83_1, reinterpret_tensor(buf100, (1568, 768), (768, 1), 0), reinterpret_tensor(arg82_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf101)
    del arg82_1
    del arg83_1
    buf102 = reinterpret_tensor(buf101, (8, 196, 3072), (602112, 3072, 1), 0); del buf101  # reuse
    cpp_fused_gelu_28(c_void_p(buf102.data_ptr()))
    buf103 = reinterpret_tensor(buf100, (1568, 768), (768, 1), 0); del buf100  # reuse
    # Source Nodes: [x_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf102, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg84_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf103)
    del arg84_1
    del arg85_1
    buf104 = buf98; del buf98  # reuse
    buf105 = buf97; del buf97  # reuse
    buf107 = reinterpret_tensor(buf74, (8, 768, 196), (150528, 196, 1), 0); del buf74  # reuse
    cpp_fused_add_clone_native_layer_norm_29(c_void_p(buf89.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf107.data_ptr()))
    del arg86_1
    del arg87_1
    buf108 = reinterpret_tensor(buf95, (6144, 384), (384, 1), 0); del buf95  # reuse
    # Source Nodes: [x_102], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (6144, 196), (196, 1), 0), reinterpret_tensor(arg88_1, (196, 384), (1, 196), 0), out=buf108)
    del arg88_1
    buf109 = reinterpret_tensor(buf108, (8, 768, 384), (294912, 384, 1), 0); del buf108  # reuse
    cpp_fused_add_gelu_30(c_void_p(buf109.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg89_1
    buf110 = reinterpret_tensor(buf107, (6144, 196), (196, 1), 0); del buf107  # reuse
    # Source Nodes: [x_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf109, (6144, 384), (384, 1), 0), reinterpret_tensor(arg90_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf110)
    del arg90_1
    del arg91_1
    buf111 = buf105; del buf105  # reuse
    buf112 = buf104; del buf104  # reuse
    buf114 = reinterpret_tensor(buf67, (8, 196, 768), (150528, 768, 1), 0); del buf67  # reuse
    cpp_fused_add_native_layer_norm_31(c_void_p(buf89.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()))
    del arg92_1
    del arg93_1
    buf115 = reinterpret_tensor(buf102, (1568, 3072), (3072, 1), 0); del buf102  # reuse
    # Source Nodes: [x_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, reinterpret_tensor(buf114, (1568, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf115)
    del arg94_1
    del arg95_1
    buf116 = reinterpret_tensor(buf115, (8, 196, 3072), (602112, 3072, 1), 0); del buf115  # reuse
    cpp_fused_gelu_32(c_void_p(buf116.data_ptr()))
    buf117 = reinterpret_tensor(buf114, (1568, 768), (768, 1), 0); del buf114  # reuse
    # Source Nodes: [x_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf116, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg96_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf117)
    del arg96_1
    del arg97_1
    buf118 = reinterpret_tensor(buf117, (8, 196, 768), (150528, 768, 1), 0); del buf117  # reuse
    buf119 = buf112; del buf112  # reuse
    buf120 = buf111; del buf111  # reuse
    buf122 = reinterpret_tensor(buf60, (8, 768, 196), (150528, 196, 1), 0); del buf60  # reuse
    cpp_fused_add_clone_native_layer_norm_33(c_void_p(buf118.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()))
    del arg98_1
    del arg99_1
    buf123 = reinterpret_tensor(buf109, (6144, 384), (384, 1), 0); del buf109  # reuse
    # Source Nodes: [x_116], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf122, (6144, 196), (196, 1), 0), reinterpret_tensor(arg100_1, (196, 384), (1, 196), 0), out=buf123)
    del arg100_1
    buf124 = reinterpret_tensor(buf123, (8, 768, 384), (294912, 384, 1), 0); del buf123  # reuse
    cpp_fused_add_gelu_34(c_void_p(buf124.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg101_1
    buf125 = reinterpret_tensor(buf122, (6144, 196), (196, 1), 0); del buf122  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf124, (6144, 384), (384, 1), 0), reinterpret_tensor(arg102_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf125)
    del arg102_1
    del arg103_1
    buf126 = buf120; del buf120  # reuse
    buf127 = buf119; del buf119  # reuse
    buf129 = reinterpret_tensor(buf96, (8, 196, 768), (150528, 768, 1), 0); del buf96  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf118.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()))
    del arg104_1
    del arg105_1
    buf130 = reinterpret_tensor(buf116, (1568, 3072), (3072, 1), 0); del buf116  # reuse
    # Source Nodes: [x_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg107_1, reinterpret_tensor(buf129, (1568, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf130)
    del arg106_1
    del arg107_1
    buf131 = reinterpret_tensor(buf130, (8, 196, 3072), (602112, 3072, 1), 0); del buf130  # reuse
    cpp_fused_gelu_36(c_void_p(buf131.data_ptr()))
    buf132 = reinterpret_tensor(buf129, (1568, 768), (768, 1), 0); del buf129  # reuse
    # Source Nodes: [x_127], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg109_1, reinterpret_tensor(buf131, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg108_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf132)
    del arg108_1
    del arg109_1
    buf133 = buf127; del buf127  # reuse
    buf134 = buf126; del buf126  # reuse
    buf136 = reinterpret_tensor(buf89, (8, 768, 196), (150528, 196, 1), 0); del buf89  # reuse
    cpp_fused_add_clone_native_layer_norm_37(c_void_p(buf118.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()))
    del arg110_1
    del arg111_1
    buf137 = reinterpret_tensor(buf124, (6144, 384), (384, 1), 0); del buf124  # reuse
    # Source Nodes: [x_130], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (6144, 196), (196, 1), 0), reinterpret_tensor(arg112_1, (196, 384), (1, 196), 0), out=buf137)
    del arg112_1
    buf138 = reinterpret_tensor(buf137, (8, 768, 384), (294912, 384, 1), 0); del buf137  # reuse
    cpp_fused_add_gelu_38(c_void_p(buf138.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg113_1
    buf139 = reinterpret_tensor(buf136, (6144, 196), (196, 1), 0); del buf136  # reuse
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf138, (6144, 384), (384, 1), 0), reinterpret_tensor(arg114_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf139)
    del arg114_1
    del arg115_1
    buf140 = buf134; del buf134  # reuse
    buf141 = buf133; del buf133  # reuse
    buf143 = reinterpret_tensor(buf110, (8, 196, 768), (150528, 768, 1), 0); del buf110  # reuse
    cpp_fused_add_native_layer_norm_39(c_void_p(buf118.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()))
    del arg116_1
    del arg117_1
    buf144 = reinterpret_tensor(buf131, (1568, 3072), (3072, 1), 0); del buf131  # reuse
    # Source Nodes: [x_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf143, (1568, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf144)
    del arg118_1
    del arg119_1
    buf145 = reinterpret_tensor(buf144, (8, 196, 3072), (602112, 3072, 1), 0); del buf144  # reuse
    cpp_fused_gelu_40(c_void_p(buf145.data_ptr()))
    buf146 = reinterpret_tensor(buf143, (1568, 768), (768, 1), 0); del buf143  # reuse
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf145, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg120_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf146)
    del arg120_1
    del arg121_1
    buf147 = reinterpret_tensor(buf146, (8, 196, 768), (150528, 768, 1), 0); del buf146  # reuse
    buf148 = buf141; del buf141  # reuse
    buf149 = buf140; del buf140  # reuse
    buf151 = reinterpret_tensor(buf103, (8, 768, 196), (150528, 196, 1), 0); del buf103  # reuse
    cpp_fused_add_clone_native_layer_norm_41(c_void_p(buf147.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()))
    del arg122_1
    del arg123_1
    del buf118
    buf152 = reinterpret_tensor(buf138, (6144, 384), (384, 1), 0); del buf138  # reuse
    # Source Nodes: [x_144], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf151, (6144, 196), (196, 1), 0), reinterpret_tensor(arg124_1, (196, 384), (1, 196), 0), out=buf152)
    del arg124_1
    buf153 = reinterpret_tensor(buf152, (8, 768, 384), (294912, 384, 1), 0); del buf152  # reuse
    cpp_fused_add_gelu_42(c_void_p(buf153.data_ptr()), c_void_p(arg125_1.data_ptr()))
    del arg125_1
    buf154 = reinterpret_tensor(buf151, (6144, 196), (196, 1), 0); del buf151  # reuse
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg127_1, reinterpret_tensor(buf153, (6144, 384), (384, 1), 0), reinterpret_tensor(arg126_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf154)
    del arg126_1
    del arg127_1
    buf155 = buf149; del buf149  # reuse
    buf156 = buf148; del buf148  # reuse
    buf158 = reinterpret_tensor(buf139, (8, 196, 768), (150528, 768, 1), 0); del buf139  # reuse
    cpp_fused_add_native_layer_norm_43(c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()))
    del arg128_1
    del arg129_1
    buf159 = reinterpret_tensor(buf145, (1568, 3072), (3072, 1), 0); del buf145  # reuse
    # Source Nodes: [x_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf158, (1568, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf159)
    del arg130_1
    del arg131_1
    buf160 = reinterpret_tensor(buf159, (8, 196, 3072), (602112, 3072, 1), 0); del buf159  # reuse
    cpp_fused_gelu_44(c_void_p(buf160.data_ptr()))
    buf161 = reinterpret_tensor(buf158, (1568, 768), (768, 1), 0); del buf158  # reuse
    # Source Nodes: [x_155], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg133_1, reinterpret_tensor(buf160, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg132_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf161)
    del arg132_1
    del arg133_1
    buf162 = buf156; del buf156  # reuse
    buf163 = buf155; del buf155  # reuse
    buf165 = reinterpret_tensor(buf132, (8, 768, 196), (150528, 196, 1), 0); del buf132  # reuse
    cpp_fused_add_clone_native_layer_norm_45(c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()))
    del arg134_1
    del arg135_1
    buf166 = reinterpret_tensor(buf153, (6144, 384), (384, 1), 0); del buf153  # reuse
    # Source Nodes: [x_158], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf165, (6144, 196), (196, 1), 0), reinterpret_tensor(arg136_1, (196, 384), (1, 196), 0), out=buf166)
    del arg136_1
    buf167 = reinterpret_tensor(buf166, (8, 768, 384), (294912, 384, 1), 0); del buf166  # reuse
    cpp_fused_add_gelu_46(c_void_p(buf167.data_ptr()), c_void_p(arg137_1.data_ptr()))
    del arg137_1
    buf168 = reinterpret_tensor(buf165, (6144, 196), (196, 1), 0); del buf165  # reuse
    # Source Nodes: [x_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg139_1, reinterpret_tensor(buf167, (6144, 384), (384, 1), 0), reinterpret_tensor(arg138_1, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf168)
    del arg138_1
    del arg139_1
    del buf167
    buf169 = buf163; del buf163  # reuse
    buf170 = buf162; del buf162  # reuse
    buf172 = reinterpret_tensor(buf125, (8, 196, 768), (150528, 768, 1), 0); del buf125  # reuse
    cpp_fused_add_native_layer_norm_47(c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()))
    del arg140_1
    del arg141_1
    buf173 = reinterpret_tensor(buf160, (1568, 3072), (3072, 1), 0); del buf160  # reuse
    # Source Nodes: [x_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf172, (1568, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf173)
    del arg142_1
    del arg143_1
    buf174 = reinterpret_tensor(buf173, (8, 196, 3072), (602112, 3072, 1), 0); del buf173  # reuse
    cpp_fused_gelu_48(c_void_p(buf174.data_ptr()))
    buf175 = reinterpret_tensor(buf172, (1568, 768), (768, 1), 0); del buf172  # reuse
    # Source Nodes: [x_169], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf174, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg144_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf175)
    del arg144_1
    del arg145_1
    del buf174
    buf176 = reinterpret_tensor(buf175, (8, 196, 768), (150528, 768, 1), 0); del buf175  # reuse
    buf177 = buf170; del buf170  # reuse
    buf178 = buf169; del buf169  # reuse
    buf180 = empty((8, 768), device='cpu', dtype=torch.float32)
    buf181 = buf180; del buf180  # reuse
    cpp_fused_add_mean_native_layer_norm_49(c_void_p(buf176.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    del arg146_1
    del arg147_1
    del buf147
    del buf154
    del buf161
    del buf168
    del buf176
    del buf177
    del buf178
    buf182 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_174, x_175, x_177], Original ATen: [aten.addmm, aten.mean, aten.native_layer_norm]
    extern_kernels.addmm(arg149_1, buf181, reinterpret_tensor(arg148_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf182)
    del arg148_1
    del arg149_1
    return (buf182, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mixer_b16_224', benchmark_compiled_module)
