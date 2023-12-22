
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
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
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (768L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (768L*x0))] = tmp0;
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
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_native_layer_norm_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr2[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr2[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_5 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(768.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_7 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_11 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_13 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(768.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_15 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_21 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(768.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_23 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_27 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_29 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(768.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_37 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(768.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_39 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_43 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(768.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_45 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(768.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(768.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((768L*x1) + (768L*x1_inner) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((768L*x1) + (150528L*(c10::div_floor_integer(x0, 768L))) + (static_cast<long>(x0) % static_cast<long>(768L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(768L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_47 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr1;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_native_layer_norm_backward_50 = async_compile.cpp('''
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
                       float* in_out_ptr24)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr7 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr9 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr12 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr15 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr18 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr21 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr24 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151 = args
    args.clear()
    assert_size_stride(primals_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (384, 196), (196, 1))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_7, (196, 384), (384, 1))
    assert_size_stride(primals_8, (196, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (3072, 768), (768, 1))
    assert_size_stride(primals_12, (3072, ), (1, ))
    assert_size_stride(primals_13, (768, 3072), (3072, 1))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (384, 196), (196, 1))
    assert_size_stride(primals_18, (384, ), (1, ))
    assert_size_stride(primals_19, (196, 384), (384, 1))
    assert_size_stride(primals_20, (196, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (3072, 768), (768, 1))
    assert_size_stride(primals_24, (3072, ), (1, ))
    assert_size_stride(primals_25, (768, 3072), (3072, 1))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (384, 196), (196, 1))
    assert_size_stride(primals_30, (384, ), (1, ))
    assert_size_stride(primals_31, (196, 384), (384, 1))
    assert_size_stride(primals_32, (196, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (3072, 768), (768, 1))
    assert_size_stride(primals_36, (3072, ), (1, ))
    assert_size_stride(primals_37, (768, 3072), (3072, 1))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (384, 196), (196, 1))
    assert_size_stride(primals_42, (384, ), (1, ))
    assert_size_stride(primals_43, (196, 384), (384, 1))
    assert_size_stride(primals_44, (196, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (3072, 768), (768, 1))
    assert_size_stride(primals_48, (3072, ), (1, ))
    assert_size_stride(primals_49, (768, 3072), (3072, 1))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (384, 196), (196, 1))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_55, (196, 384), (384, 1))
    assert_size_stride(primals_56, (196, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (3072, 768), (768, 1))
    assert_size_stride(primals_60, (3072, ), (1, ))
    assert_size_stride(primals_61, (768, 3072), (3072, 1))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (384, 196), (196, 1))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (196, 384), (384, 1))
    assert_size_stride(primals_68, (196, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (3072, 768), (768, 1))
    assert_size_stride(primals_72, (3072, ), (1, ))
    assert_size_stride(primals_73, (768, 3072), (3072, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (384, 196), (196, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (196, 384), (384, 1))
    assert_size_stride(primals_80, (196, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (3072, 768), (768, 1))
    assert_size_stride(primals_84, (3072, ), (1, ))
    assert_size_stride(primals_85, (768, 3072), (3072, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (384, 196), (196, 1))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (196, 384), (384, 1))
    assert_size_stride(primals_92, (196, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (3072, 768), (768, 1))
    assert_size_stride(primals_96, (3072, ), (1, ))
    assert_size_stride(primals_97, (768, 3072), (3072, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (384, 196), (196, 1))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (196, 384), (384, 1))
    assert_size_stride(primals_104, (196, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (3072, 768), (768, 1))
    assert_size_stride(primals_108, (3072, ), (1, ))
    assert_size_stride(primals_109, (768, 3072), (3072, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (384, 196), (196, 1))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_115, (196, 384), (384, 1))
    assert_size_stride(primals_116, (196, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (3072, 768), (768, 1))
    assert_size_stride(primals_120, (3072, ), (1, ))
    assert_size_stride(primals_121, (768, 3072), (3072, 1))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (384, 196), (196, 1))
    assert_size_stride(primals_126, (384, ), (1, ))
    assert_size_stride(primals_127, (196, 384), (384, 1))
    assert_size_stride(primals_128, (196, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (3072, 768), (768, 1))
    assert_size_stride(primals_132, (3072, ), (1, ))
    assert_size_stride(primals_133, (768, 3072), (3072, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (384, 196), (196, 1))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (196, 384), (384, 1))
    assert_size_stride(primals_140, (196, ), (1, ))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (3072, 768), (768, 1))
    assert_size_stride(primals_144, (3072, ), (1, ))
    assert_size_stride(primals_145, (768, 3072), (3072, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (1000, 768), (768, 1))
    assert_size_stride(primals_150, (1000, ), (1, ))
    assert_size_stride(primals_151, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((768, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_1
    del primals_151
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, primals_2, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 768, 14, 14), (150528, 1, 10752, 768))
    del primals_2
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf7 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del primals_4
    buf8 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_4], Original ATen: [aten.mm]
    extern_kernels.mm(buf7, reinterpret_tensor(primals_5, (196, 384), (1, 196), 0), out=buf8)
    buf9 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_2(c_void_p(buf8.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf9.data_ptr()))
    buf10 = empty((6144, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, buf9, reinterpret_tensor(primals_7, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf10)
    del primals_8
    buf11 = buf3; del buf3  # reuse
    buf12 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf14 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf15 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_3(c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    del primals_10
    buf16 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_12, buf15, reinterpret_tensor(primals_11, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf16)
    del primals_12
    buf17 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_4(c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    buf18 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_14, buf17, reinterpret_tensor(primals_13, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf18)
    del primals_14
    buf19 = buf11; del buf11  # reuse
    buf20 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf22 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf23 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_5(c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    del primals_16
    buf24 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_18], Original ATen: [aten.mm]
    extern_kernels.mm(buf23, reinterpret_tensor(primals_17, (196, 384), (1, 196), 0), out=buf24)
    buf25 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_6(c_void_p(buf24.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf25.data_ptr()))
    buf26 = empty((6144, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_20, buf25, reinterpret_tensor(primals_19, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf26)
    del primals_20
    buf27 = buf19; del buf19  # reuse
    buf28 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf30 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf31 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_7(c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del primals_22
    buf32 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_24, buf31, reinterpret_tensor(primals_23, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf32)
    del primals_24
    buf33 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_8(c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    buf34 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_26, buf33, reinterpret_tensor(primals_25, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf34)
    del primals_26
    buf35 = reinterpret_tensor(buf34, (8, 196, 768), (150528, 768, 1), 0); del buf34  # reuse
    buf36 = buf27; del buf27  # reuse
    buf37 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf39 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf40 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_9(c_void_p(buf35.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    del primals_28
    buf41 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_32], Original ATen: [aten.mm]
    extern_kernels.mm(buf40, reinterpret_tensor(primals_29, (196, 384), (1, 196), 0), out=buf41)
    buf42 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_10(c_void_p(buf41.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf42.data_ptr()))
    buf43 = buf26; del buf26  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_32, buf42, reinterpret_tensor(primals_31, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf43)
    del primals_32
    buf44 = buf36; del buf36  # reuse
    buf45 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf47 = reinterpret_tensor(buf2, (8, 196, 768), (150528, 768, 1), 0); del buf2  # reuse
    buf48 = buf18; del buf18  # reuse
    cpp_fused_add_native_layer_norm_view_11(c_void_p(buf35.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del primals_34
    buf49 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_36, buf48, reinterpret_tensor(primals_35, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf49)
    del primals_36
    buf50 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_12(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    buf51 = reinterpret_tensor(buf10, (1568, 768), (768, 1), 0); del buf10  # reuse
    # Source Nodes: [x_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_38, buf50, reinterpret_tensor(primals_37, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf51)
    del primals_38
    buf52 = buf44; del buf44  # reuse
    buf53 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf55 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf56 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_13(c_void_p(buf35.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    del primals_40
    buf57 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_46], Original ATen: [aten.mm]
    extern_kernels.mm(buf56, reinterpret_tensor(primals_41, (196, 384), (1, 196), 0), out=buf57)
    buf58 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_14(c_void_p(buf57.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf58.data_ptr()))
    buf59 = empty((6144, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_44, buf58, reinterpret_tensor(primals_43, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf59)
    del primals_44
    buf60 = buf52; del buf52  # reuse
    buf61 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf63 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf64 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_15(c_void_p(buf35.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()))
    del primals_46
    buf65 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_48, buf64, reinterpret_tensor(primals_47, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf65)
    del primals_48
    buf66 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_16(c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    buf67 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_50, buf66, reinterpret_tensor(primals_49, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf67)
    del primals_50
    buf68 = reinterpret_tensor(buf67, (8, 196, 768), (150528, 768, 1), 0); del buf67  # reuse
    buf69 = buf60; del buf60  # reuse
    buf70 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf72 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf73 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_17(c_void_p(buf68.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_52
    buf74 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_60], Original ATen: [aten.mm]
    extern_kernels.mm(buf73, reinterpret_tensor(primals_53, (196, 384), (1, 196), 0), out=buf74)
    buf75 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_18(c_void_p(buf74.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = buf59; del buf59  # reuse
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_56, buf75, reinterpret_tensor(primals_55, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf76)
    del primals_56
    buf77 = buf69; del buf69  # reuse
    buf78 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf80 = reinterpret_tensor(buf51, (8, 196, 768), (150528, 768, 1), 0); del buf51  # reuse
    buf81 = reinterpret_tensor(buf43, (1568, 768), (768, 1), 0); del buf43  # reuse
    cpp_fused_add_native_layer_norm_view_19(c_void_p(buf68.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del primals_58
    buf82 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, buf81, reinterpret_tensor(primals_59, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf82)
    del primals_60
    buf83 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_20(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    buf84 = reinterpret_tensor(buf35, (1568, 768), (768, 1), 0); del buf35  # reuse
    # Source Nodes: [x_71], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_62, buf83, reinterpret_tensor(primals_61, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf84)
    del primals_62
    buf85 = buf77; del buf77  # reuse
    buf86 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf88 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf89 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_21(c_void_p(buf68.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    del primals_64
    buf90 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_74], Original ATen: [aten.mm]
    extern_kernels.mm(buf89, reinterpret_tensor(primals_65, (196, 384), (1, 196), 0), out=buf90)
    buf91 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_22(c_void_p(buf90.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf91.data_ptr()))
    buf92 = empty((6144, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_68, buf91, reinterpret_tensor(primals_67, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf92)
    del primals_68
    buf93 = buf85; del buf85  # reuse
    buf94 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf96 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf97 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_23(c_void_p(buf68.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_70
    buf98 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_72, buf97, reinterpret_tensor(primals_71, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf98)
    del primals_72
    buf99 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_24(c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    buf100 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, buf99, reinterpret_tensor(primals_73, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf100)
    del primals_74
    buf101 = reinterpret_tensor(buf100, (8, 196, 768), (150528, 768, 1), 0); del buf100  # reuse
    buf102 = buf93; del buf93  # reuse
    buf103 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf105 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf106 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_25(c_void_p(buf101.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del primals_76
    buf107 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_88], Original ATen: [aten.mm]
    extern_kernels.mm(buf106, reinterpret_tensor(primals_77, (196, 384), (1, 196), 0), out=buf107)
    buf108 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_26(c_void_p(buf107.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf108.data_ptr()))
    buf109 = buf92; del buf92  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_80, buf108, reinterpret_tensor(primals_79, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf109)
    del primals_80
    buf110 = buf102; del buf102  # reuse
    buf111 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf113 = reinterpret_tensor(buf84, (8, 196, 768), (150528, 768, 1), 0); del buf84  # reuse
    buf114 = reinterpret_tensor(buf76, (1568, 768), (768, 1), 0); del buf76  # reuse
    cpp_fused_add_native_layer_norm_view_27(c_void_p(buf101.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()))
    del primals_82
    buf115 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_84, buf114, reinterpret_tensor(primals_83, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf115)
    del primals_84
    buf116 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_28(c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    buf117 = reinterpret_tensor(buf68, (1568, 768), (768, 1), 0); del buf68  # reuse
    # Source Nodes: [x_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, buf116, reinterpret_tensor(primals_85, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf117)
    del primals_86
    buf118 = buf110; del buf110  # reuse
    buf119 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf121 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf122 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_29(c_void_p(buf101.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    del primals_88
    buf123 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_102], Original ATen: [aten.mm]
    extern_kernels.mm(buf122, reinterpret_tensor(primals_89, (196, 384), (1, 196), 0), out=buf123)
    buf124 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_30(c_void_p(buf123.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf124.data_ptr()))
    buf125 = empty((6144, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf124, reinterpret_tensor(primals_91, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf125)
    del primals_92
    buf126 = buf118; del buf118  # reuse
    buf127 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf129 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf130 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_31(c_void_p(buf101.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    del primals_94
    buf131 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_96, buf130, reinterpret_tensor(primals_95, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf131)
    del primals_96
    buf132 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_32(c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    buf133 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_98, buf132, reinterpret_tensor(primals_97, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf133)
    del primals_98
    buf134 = reinterpret_tensor(buf133, (8, 196, 768), (150528, 768, 1), 0); del buf133  # reuse
    buf135 = buf126; del buf126  # reuse
    buf136 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf138 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf139 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_33(c_void_p(buf134.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del primals_100
    buf140 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_116], Original ATen: [aten.mm]
    extern_kernels.mm(buf139, reinterpret_tensor(primals_101, (196, 384), (1, 196), 0), out=buf140)
    buf141 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_34(c_void_p(buf140.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf141.data_ptr()))
    buf142 = buf125; del buf125  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf141, reinterpret_tensor(primals_103, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf142)
    del primals_104
    buf143 = buf135; del buf135  # reuse
    buf144 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf146 = reinterpret_tensor(buf117, (8, 196, 768), (150528, 768, 1), 0); del buf117  # reuse
    buf147 = reinterpret_tensor(buf109, (1568, 768), (768, 1), 0); del buf109  # reuse
    cpp_fused_add_native_layer_norm_view_35(c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    del primals_106
    buf148 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_108, buf147, reinterpret_tensor(primals_107, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf148)
    del primals_108
    buf149 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_36(c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()))
    buf150 = reinterpret_tensor(buf101, (1568, 768), (768, 1), 0); del buf101  # reuse
    # Source Nodes: [x_127], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_110, buf149, reinterpret_tensor(primals_109, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf150)
    del primals_110
    buf151 = buf143; del buf143  # reuse
    buf152 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf154 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf155 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_37(c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    del primals_112
    buf156 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_130], Original ATen: [aten.mm]
    extern_kernels.mm(buf155, reinterpret_tensor(primals_113, (196, 384), (1, 196), 0), out=buf156)
    buf157 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_38(c_void_p(buf156.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf157.data_ptr()))
    buf158 = empty((6144, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_116, buf157, reinterpret_tensor(primals_115, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf158)
    del primals_116
    buf159 = buf151; del buf151  # reuse
    buf160 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf162 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf163 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_39(c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    del primals_118
    buf164 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf163, reinterpret_tensor(primals_119, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf164)
    del primals_120
    buf165 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_40(c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    buf166 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_122, buf165, reinterpret_tensor(primals_121, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf166)
    del primals_122
    buf167 = reinterpret_tensor(buf166, (8, 196, 768), (150528, 768, 1), 0); del buf166  # reuse
    buf168 = buf159; del buf159  # reuse
    buf169 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf171 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf172 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_41(c_void_p(buf167.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del primals_124
    buf173 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_144], Original ATen: [aten.mm]
    extern_kernels.mm(buf172, reinterpret_tensor(primals_125, (196, 384), (1, 196), 0), out=buf173)
    buf174 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_42(c_void_p(buf173.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf174.data_ptr()))
    buf175 = buf158; del buf158  # reuse
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_128, buf174, reinterpret_tensor(primals_127, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf175)
    del primals_128
    buf176 = buf168; del buf168  # reuse
    buf177 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf179 = reinterpret_tensor(buf150, (8, 196, 768), (150528, 768, 1), 0); del buf150  # reuse
    buf180 = reinterpret_tensor(buf142, (1568, 768), (768, 1), 0); del buf142  # reuse
    cpp_fused_add_native_layer_norm_view_43(c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()))
    del primals_130
    buf181 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf180, reinterpret_tensor(primals_131, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf181)
    del primals_132
    buf182 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_44(c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()))
    buf183 = reinterpret_tensor(buf134, (1568, 768), (768, 1), 0); del buf134  # reuse
    # Source Nodes: [x_155], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_134, buf182, reinterpret_tensor(primals_133, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf183)
    del primals_134
    buf184 = buf176; del buf176  # reuse
    buf185 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf187 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf188 = empty((6144, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_45(c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    del primals_136
    buf189 = empty((6144, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_158], Original ATen: [aten.mm]
    extern_kernels.mm(buf188, reinterpret_tensor(primals_137, (196, 384), (1, 196), 0), out=buf189)
    buf190 = empty((6144, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_view_46(c_void_p(buf189.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = empty((6144, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_140, buf190, reinterpret_tensor(primals_139, (384, 196), (1, 384), 0), alpha=1, beta=1, out=buf191)
    del primals_140
    buf192 = buf184; del buf184  # reuse
    buf193 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf195 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf196 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_47(c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()))
    del primals_142
    buf197 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf196, reinterpret_tensor(primals_143, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf197)
    del primals_144
    buf198 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_48(c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    buf199 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_169], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_146, buf198, reinterpret_tensor(primals_145, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf199)
    del primals_146
    buf200 = reinterpret_tensor(buf199, (8, 196, 768), (150528, 768, 1), 0); del buf199  # reuse
    buf201 = buf192; del buf192  # reuse
    buf202 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf204 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf205 = empty((8, 768), device='cpu', dtype=torch.float32)
    buf206 = buf205; del buf205  # reuse
    cpp_fused_add_mean_native_layer_norm_49(c_void_p(buf200.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()))
    del buf167
    del buf175
    del buf183
    del buf191
    del buf200
    del buf201
    del primals_148
    buf207 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_150, buf206, reinterpret_tensor(primals_149, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf207)
    del primals_150
    buf208 = reinterpret_tensor(buf202, (8, 196, 1), (196, 1, 1), 0); del buf202  # reuse
    buf209 = reinterpret_tensor(buf193, (8, 196, 1), (196, 1, 1), 0); del buf193  # reuse
    buf210 = reinterpret_tensor(buf185, (8, 196, 1), (196, 1, 1), 0); del buf185  # reuse
    buf211 = reinterpret_tensor(buf177, (8, 196, 1), (196, 1, 1), 0); del buf177  # reuse
    buf212 = reinterpret_tensor(buf169, (8, 196, 1), (196, 1, 1), 0); del buf169  # reuse
    buf213 = reinterpret_tensor(buf160, (8, 196, 1), (196, 1, 1), 0); del buf160  # reuse
    buf214 = reinterpret_tensor(buf152, (8, 196, 1), (196, 1, 1), 0); del buf152  # reuse
    buf215 = reinterpret_tensor(buf144, (8, 196, 1), (196, 1, 1), 0); del buf144  # reuse
    buf216 = reinterpret_tensor(buf136, (8, 196, 1), (196, 1, 1), 0); del buf136  # reuse
    buf217 = reinterpret_tensor(buf127, (8, 196, 1), (196, 1, 1), 0); del buf127  # reuse
    buf218 = reinterpret_tensor(buf119, (8, 196, 1), (196, 1, 1), 0); del buf119  # reuse
    buf219 = reinterpret_tensor(buf111, (8, 196, 1), (196, 1, 1), 0); del buf111  # reuse
    buf220 = reinterpret_tensor(buf103, (8, 196, 1), (196, 1, 1), 0); del buf103  # reuse
    buf221 = reinterpret_tensor(buf94, (8, 196, 1), (196, 1, 1), 0); del buf94  # reuse
    buf222 = reinterpret_tensor(buf86, (8, 196, 1), (196, 1, 1), 0); del buf86  # reuse
    buf223 = reinterpret_tensor(buf78, (8, 196, 1), (196, 1, 1), 0); del buf78  # reuse
    buf224 = reinterpret_tensor(buf70, (8, 196, 1), (196, 1, 1), 0); del buf70  # reuse
    buf225 = reinterpret_tensor(buf61, (8, 196, 1), (196, 1, 1), 0); del buf61  # reuse
    buf226 = reinterpret_tensor(buf53, (8, 196, 1), (196, 1, 1), 0); del buf53  # reuse
    buf227 = reinterpret_tensor(buf45, (8, 196, 1), (196, 1, 1), 0); del buf45  # reuse
    buf228 = reinterpret_tensor(buf37, (8, 196, 1), (196, 1, 1), 0); del buf37  # reuse
    buf229 = reinterpret_tensor(buf28, (8, 196, 1), (196, 1, 1), 0); del buf28  # reuse
    buf230 = reinterpret_tensor(buf20, (8, 196, 1), (196, 1, 1), 0); del buf20  # reuse
    buf231 = reinterpret_tensor(buf12, (8, 196, 1), (196, 1, 1), 0); del buf12  # reuse
    buf232 = reinterpret_tensor(buf4, (8, 196, 1), (196, 1, 1), 0); del buf4  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_50(c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    return (buf207, buf0, primals_3, primals_6, primals_9, primals_15, primals_18, primals_21, primals_27, primals_30, primals_33, primals_39, primals_42, primals_45, primals_51, primals_54, primals_57, primals_63, primals_66, primals_69, primals_75, primals_78, primals_81, primals_87, primals_90, primals_93, primals_99, primals_102, primals_105, primals_111, primals_114, primals_117, primals_123, primals_126, primals_129, primals_135, primals_138, primals_141, primals_147, buf1, buf6, buf7, buf8, buf9, buf14, buf15, buf16, buf17, buf22, buf23, buf24, buf25, buf30, buf31, buf32, buf33, buf39, buf40, buf41, buf42, buf47, buf48, buf49, buf50, buf55, buf56, buf57, buf58, buf63, buf64, buf65, buf66, buf72, buf73, buf74, buf75, buf80, buf81, buf82, buf83, buf88, buf89, buf90, buf91, buf96, buf97, buf98, buf99, buf105, buf106, buf107, buf108, buf113, buf114, buf115, buf116, buf121, buf122, buf123, buf124, buf129, buf130, buf131, buf132, buf138, buf139, buf140, buf141, buf146, buf147, buf148, buf149, buf154, buf155, buf156, buf157, buf162, buf163, buf164, buf165, buf171, buf172, buf173, buf174, buf179, buf180, buf181, buf182, buf187, buf188, buf189, buf190, buf195, buf196, buf197, buf198, buf204, buf206, reinterpret_tensor(primals_149, (1000, 768), (768, 1), 0), buf208, reinterpret_tensor(primals_145, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_143, (3072, 768), (768, 1), 0), buf209, reinterpret_tensor(primals_139, (196, 384), (384, 1), 0), reinterpret_tensor(primals_137, (384, 196), (196, 1), 0), buf210, reinterpret_tensor(primals_133, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_131, (3072, 768), (768, 1), 0), buf211, reinterpret_tensor(primals_127, (196, 384), (384, 1), 0), reinterpret_tensor(primals_125, (384, 196), (196, 1), 0), buf212, reinterpret_tensor(primals_121, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_119, (3072, 768), (768, 1), 0), buf213, reinterpret_tensor(primals_115, (196, 384), (384, 1), 0), reinterpret_tensor(primals_113, (384, 196), (196, 1), 0), buf214, reinterpret_tensor(primals_109, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_107, (3072, 768), (768, 1), 0), buf215, reinterpret_tensor(primals_103, (196, 384), (384, 1), 0), reinterpret_tensor(primals_101, (384, 196), (196, 1), 0), buf216, reinterpret_tensor(primals_97, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_95, (3072, 768), (768, 1), 0), buf217, reinterpret_tensor(primals_91, (196, 384), (384, 1), 0), reinterpret_tensor(primals_89, (384, 196), (196, 1), 0), buf218, reinterpret_tensor(primals_85, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_83, (3072, 768), (768, 1), 0), buf219, reinterpret_tensor(primals_79, (196, 384), (384, 1), 0), reinterpret_tensor(primals_77, (384, 196), (196, 1), 0), buf220, reinterpret_tensor(primals_73, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_71, (3072, 768), (768, 1), 0), buf221, reinterpret_tensor(primals_67, (196, 384), (384, 1), 0), reinterpret_tensor(primals_65, (384, 196), (196, 1), 0), buf222, reinterpret_tensor(primals_61, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_59, (3072, 768), (768, 1), 0), buf223, reinterpret_tensor(primals_55, (196, 384), (384, 1), 0), reinterpret_tensor(primals_53, (384, 196), (196, 1), 0), buf224, reinterpret_tensor(primals_49, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_47, (3072, 768), (768, 1), 0), buf225, reinterpret_tensor(primals_43, (196, 384), (384, 1), 0), reinterpret_tensor(primals_41, (384, 196), (196, 1), 0), buf226, reinterpret_tensor(primals_37, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_35, (3072, 768), (768, 1), 0), buf227, reinterpret_tensor(primals_31, (196, 384), (384, 1), 0), reinterpret_tensor(primals_29, (384, 196), (196, 1), 0), buf228, reinterpret_tensor(primals_25, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_23, (3072, 768), (768, 1), 0), buf229, reinterpret_tensor(primals_19, (196, 384), (384, 1), 0), reinterpret_tensor(primals_17, (384, 196), (196, 1), 0), buf230, reinterpret_tensor(primals_13, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_11, (3072, 768), (768, 1), 0), buf231, reinterpret_tensor(primals_7, (196, 384), (384, 1), 0), reinterpret_tensor(primals_5, (384, 196), (196, 1), 0), buf232, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((196, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mixer_b16_224', benchmark_compiled_module)
