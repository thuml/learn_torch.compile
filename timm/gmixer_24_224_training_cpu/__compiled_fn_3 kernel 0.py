
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr2[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr2[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_2 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_4 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_6 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_10 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_12 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_14 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_18 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_20 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_22 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_26 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_28 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_30 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_34 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_36 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_38 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_42 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_44 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_46 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_48 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_50 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_52 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_53 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_54 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_55 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_56 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_58 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_59 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_60 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_61 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_62 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_63 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_64 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_66 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_67 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_68 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_69 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_70 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_71 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_72 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_74 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_75 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_76 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_77 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_78 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_79 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_80 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_82 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_83 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_84 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_85 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_86 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_87 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_88 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr5[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_90 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_91 = async_compile.cpp('''
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
                            tmp14.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(384.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp11;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_92 = async_compile.cpp('''
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


cpp_fused__unsafe_view_add_clone_native_layer_norm_93 = async_compile.cpp('''
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
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(384.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            tmp16.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(384.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp13;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp4 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    auto tmp1 = in_ptr3[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp3 = in_ptr4[static_cast<long>(static_cast<long>(x0) % static_cast<long>(384L))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_94 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_95 = async_compile.cpp('''
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
                            tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
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
                        out_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp15;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_view_96 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x2) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_native_layer_norm_backward_98 = async_compile.cpp('''
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
                       const float* in_ptr23)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr27 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr30 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr33 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr36 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr39 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr42 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr45 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr48 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr49 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr50 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr51 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr52 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr53 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr54 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr55 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr55 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr56 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr57 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr57 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr58 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr58 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr59 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr59 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr60 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr60 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr61 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr61 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr62 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr62 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr63 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr63 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr64 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr64 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr65 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr65 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr66 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr66 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr67 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr67 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr68 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr68 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr69 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr69 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr70 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr70 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr71 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr71 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr72 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr72 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295 = args
    args.clear()
    assert_size_stride(primals_1, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_2, (384, ), (1, ))
    assert_size_stride(primals_3, (384, ), (1, ))
    assert_size_stride(primals_4, (384, ), (1, ))
    assert_size_stride(primals_5, (384, 196), (196, 1))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_7, (196, 192), (192, 1))
    assert_size_stride(primals_8, (196, ), (1, ))
    assert_size_stride(primals_9, (384, ), (1, ))
    assert_size_stride(primals_10, (384, ), (1, ))
    assert_size_stride(primals_11, (1536, 384), (384, 1))
    assert_size_stride(primals_12, (1536, ), (1, ))
    assert_size_stride(primals_13, (384, 768), (768, 1))
    assert_size_stride(primals_14, (384, ), (1, ))
    assert_size_stride(primals_15, (384, ), (1, ))
    assert_size_stride(primals_16, (384, ), (1, ))
    assert_size_stride(primals_17, (384, 196), (196, 1))
    assert_size_stride(primals_18, (384, ), (1, ))
    assert_size_stride(primals_19, (196, 192), (192, 1))
    assert_size_stride(primals_20, (196, ), (1, ))
    assert_size_stride(primals_21, (384, ), (1, ))
    assert_size_stride(primals_22, (384, ), (1, ))
    assert_size_stride(primals_23, (1536, 384), (384, 1))
    assert_size_stride(primals_24, (1536, ), (1, ))
    assert_size_stride(primals_25, (384, 768), (768, 1))
    assert_size_stride(primals_26, (384, ), (1, ))
    assert_size_stride(primals_27, (384, ), (1, ))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_29, (384, 196), (196, 1))
    assert_size_stride(primals_30, (384, ), (1, ))
    assert_size_stride(primals_31, (196, 192), (192, 1))
    assert_size_stride(primals_32, (196, ), (1, ))
    assert_size_stride(primals_33, (384, ), (1, ))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_35, (1536, 384), (384, 1))
    assert_size_stride(primals_36, (1536, ), (1, ))
    assert_size_stride(primals_37, (384, 768), (768, 1))
    assert_size_stride(primals_38, (384, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_41, (384, 196), (196, 1))
    assert_size_stride(primals_42, (384, ), (1, ))
    assert_size_stride(primals_43, (196, 192), (192, 1))
    assert_size_stride(primals_44, (196, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_46, (384, ), (1, ))
    assert_size_stride(primals_47, (1536, 384), (384, 1))
    assert_size_stride(primals_48, (1536, ), (1, ))
    assert_size_stride(primals_49, (384, 768), (768, 1))
    assert_size_stride(primals_50, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_52, (384, ), (1, ))
    assert_size_stride(primals_53, (384, 196), (196, 1))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_55, (196, 192), (192, 1))
    assert_size_stride(primals_56, (196, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_59, (1536, 384), (384, 1))
    assert_size_stride(primals_60, (1536, ), (1, ))
    assert_size_stride(primals_61, (384, 768), (768, 1))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (384, ), (1, ))
    assert_size_stride(primals_65, (384, 196), (196, 1))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (196, 192), (192, 1))
    assert_size_stride(primals_68, (196, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_71, (1536, 384), (384, 1))
    assert_size_stride(primals_72, (1536, ), (1, ))
    assert_size_stride(primals_73, (384, 768), (768, 1))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (384, ), (1, ))
    assert_size_stride(primals_77, (384, 196), (196, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (196, 192), (192, 1))
    assert_size_stride(primals_80, (196, ), (1, ))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_82, (384, ), (1, ))
    assert_size_stride(primals_83, (1536, 384), (384, 1))
    assert_size_stride(primals_84, (1536, ), (1, ))
    assert_size_stride(primals_85, (384, 768), (768, 1))
    assert_size_stride(primals_86, (384, ), (1, ))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (384, ), (1, ))
    assert_size_stride(primals_89, (384, 196), (196, 1))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (196, 192), (192, 1))
    assert_size_stride(primals_92, (196, ), (1, ))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_95, (1536, 384), (384, 1))
    assert_size_stride(primals_96, (1536, ), (1, ))
    assert_size_stride(primals_97, (384, 768), (768, 1))
    assert_size_stride(primals_98, (384, ), (1, ))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_101, (384, 196), (196, 1))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (196, 192), (192, 1))
    assert_size_stride(primals_104, (196, ), (1, ))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (1536, 384), (384, 1))
    assert_size_stride(primals_108, (1536, ), (1, ))
    assert_size_stride(primals_109, (384, 768), (768, 1))
    assert_size_stride(primals_110, (384, ), (1, ))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (384, 196), (196, 1))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_115, (196, 192), (192, 1))
    assert_size_stride(primals_116, (196, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (1536, 384), (384, 1))
    assert_size_stride(primals_120, (1536, ), (1, ))
    assert_size_stride(primals_121, (384, 768), (768, 1))
    assert_size_stride(primals_122, (384, ), (1, ))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (384, ), (1, ))
    assert_size_stride(primals_125, (384, 196), (196, 1))
    assert_size_stride(primals_126, (384, ), (1, ))
    assert_size_stride(primals_127, (196, 192), (192, 1))
    assert_size_stride(primals_128, (196, ), (1, ))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (384, ), (1, ))
    assert_size_stride(primals_131, (1536, 384), (384, 1))
    assert_size_stride(primals_132, (1536, ), (1, ))
    assert_size_stride(primals_133, (384, 768), (768, 1))
    assert_size_stride(primals_134, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (384, ), (1, ))
    assert_size_stride(primals_137, (384, 196), (196, 1))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (196, 192), (192, 1))
    assert_size_stride(primals_140, (196, ), (1, ))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (384, ), (1, ))
    assert_size_stride(primals_143, (1536, 384), (384, 1))
    assert_size_stride(primals_144, (1536, ), (1, ))
    assert_size_stride(primals_145, (384, 768), (768, 1))
    assert_size_stride(primals_146, (384, ), (1, ))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_149, (384, 196), (196, 1))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (196, 192), (192, 1))
    assert_size_stride(primals_152, (196, ), (1, ))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (384, ), (1, ))
    assert_size_stride(primals_155, (1536, 384), (384, 1))
    assert_size_stride(primals_156, (1536, ), (1, ))
    assert_size_stride(primals_157, (384, 768), (768, 1))
    assert_size_stride(primals_158, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (384, ), (1, ))
    assert_size_stride(primals_161, (384, 196), (196, 1))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (196, 192), (192, 1))
    assert_size_stride(primals_164, (196, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (384, ), (1, ))
    assert_size_stride(primals_167, (1536, 384), (384, 1))
    assert_size_stride(primals_168, (1536, ), (1, ))
    assert_size_stride(primals_169, (384, 768), (768, 1))
    assert_size_stride(primals_170, (384, ), (1, ))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_172, (384, ), (1, ))
    assert_size_stride(primals_173, (384, 196), (196, 1))
    assert_size_stride(primals_174, (384, ), (1, ))
    assert_size_stride(primals_175, (196, 192), (192, 1))
    assert_size_stride(primals_176, (196, ), (1, ))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_179, (1536, 384), (384, 1))
    assert_size_stride(primals_180, (1536, ), (1, ))
    assert_size_stride(primals_181, (384, 768), (768, 1))
    assert_size_stride(primals_182, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_184, (384, ), (1, ))
    assert_size_stride(primals_185, (384, 196), (196, 1))
    assert_size_stride(primals_186, (384, ), (1, ))
    assert_size_stride(primals_187, (196, 192), (192, 1))
    assert_size_stride(primals_188, (196, ), (1, ))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_190, (384, ), (1, ))
    assert_size_stride(primals_191, (1536, 384), (384, 1))
    assert_size_stride(primals_192, (1536, ), (1, ))
    assert_size_stride(primals_193, (384, 768), (768, 1))
    assert_size_stride(primals_194, (384, ), (1, ))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (384, ), (1, ))
    assert_size_stride(primals_197, (384, 196), (196, 1))
    assert_size_stride(primals_198, (384, ), (1, ))
    assert_size_stride(primals_199, (196, 192), (192, 1))
    assert_size_stride(primals_200, (196, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (384, ), (1, ))
    assert_size_stride(primals_203, (1536, 384), (384, 1))
    assert_size_stride(primals_204, (1536, ), (1, ))
    assert_size_stride(primals_205, (384, 768), (768, 1))
    assert_size_stride(primals_206, (384, ), (1, ))
    assert_size_stride(primals_207, (384, ), (1, ))
    assert_size_stride(primals_208, (384, ), (1, ))
    assert_size_stride(primals_209, (384, 196), (196, 1))
    assert_size_stride(primals_210, (384, ), (1, ))
    assert_size_stride(primals_211, (196, 192), (192, 1))
    assert_size_stride(primals_212, (196, ), (1, ))
    assert_size_stride(primals_213, (384, ), (1, ))
    assert_size_stride(primals_214, (384, ), (1, ))
    assert_size_stride(primals_215, (1536, 384), (384, 1))
    assert_size_stride(primals_216, (1536, ), (1, ))
    assert_size_stride(primals_217, (384, 768), (768, 1))
    assert_size_stride(primals_218, (384, ), (1, ))
    assert_size_stride(primals_219, (384, ), (1, ))
    assert_size_stride(primals_220, (384, ), (1, ))
    assert_size_stride(primals_221, (384, 196), (196, 1))
    assert_size_stride(primals_222, (384, ), (1, ))
    assert_size_stride(primals_223, (196, 192), (192, 1))
    assert_size_stride(primals_224, (196, ), (1, ))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_226, (384, ), (1, ))
    assert_size_stride(primals_227, (1536, 384), (384, 1))
    assert_size_stride(primals_228, (1536, ), (1, ))
    assert_size_stride(primals_229, (384, 768), (768, 1))
    assert_size_stride(primals_230, (384, ), (1, ))
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_232, (384, ), (1, ))
    assert_size_stride(primals_233, (384, 196), (196, 1))
    assert_size_stride(primals_234, (384, ), (1, ))
    assert_size_stride(primals_235, (196, 192), (192, 1))
    assert_size_stride(primals_236, (196, ), (1, ))
    assert_size_stride(primals_237, (384, ), (1, ))
    assert_size_stride(primals_238, (384, ), (1, ))
    assert_size_stride(primals_239, (1536, 384), (384, 1))
    assert_size_stride(primals_240, (1536, ), (1, ))
    assert_size_stride(primals_241, (384, 768), (768, 1))
    assert_size_stride(primals_242, (384, ), (1, ))
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_244, (384, ), (1, ))
    assert_size_stride(primals_245, (384, 196), (196, 1))
    assert_size_stride(primals_246, (384, ), (1, ))
    assert_size_stride(primals_247, (196, 192), (192, 1))
    assert_size_stride(primals_248, (196, ), (1, ))
    assert_size_stride(primals_249, (384, ), (1, ))
    assert_size_stride(primals_250, (384, ), (1, ))
    assert_size_stride(primals_251, (1536, 384), (384, 1))
    assert_size_stride(primals_252, (1536, ), (1, ))
    assert_size_stride(primals_253, (384, 768), (768, 1))
    assert_size_stride(primals_254, (384, ), (1, ))
    assert_size_stride(primals_255, (384, ), (1, ))
    assert_size_stride(primals_256, (384, ), (1, ))
    assert_size_stride(primals_257, (384, 196), (196, 1))
    assert_size_stride(primals_258, (384, ), (1, ))
    assert_size_stride(primals_259, (196, 192), (192, 1))
    assert_size_stride(primals_260, (196, ), (1, ))
    assert_size_stride(primals_261, (384, ), (1, ))
    assert_size_stride(primals_262, (384, ), (1, ))
    assert_size_stride(primals_263, (1536, 384), (384, 1))
    assert_size_stride(primals_264, (1536, ), (1, ))
    assert_size_stride(primals_265, (384, 768), (768, 1))
    assert_size_stride(primals_266, (384, ), (1, ))
    assert_size_stride(primals_267, (384, ), (1, ))
    assert_size_stride(primals_268, (384, ), (1, ))
    assert_size_stride(primals_269, (384, 196), (196, 1))
    assert_size_stride(primals_270, (384, ), (1, ))
    assert_size_stride(primals_271, (196, 192), (192, 1))
    assert_size_stride(primals_272, (196, ), (1, ))
    assert_size_stride(primals_273, (384, ), (1, ))
    assert_size_stride(primals_274, (384, ), (1, ))
    assert_size_stride(primals_275, (1536, 384), (384, 1))
    assert_size_stride(primals_276, (1536, ), (1, ))
    assert_size_stride(primals_277, (384, 768), (768, 1))
    assert_size_stride(primals_278, (384, ), (1, ))
    assert_size_stride(primals_279, (384, ), (1, ))
    assert_size_stride(primals_280, (384, ), (1, ))
    assert_size_stride(primals_281, (384, 196), (196, 1))
    assert_size_stride(primals_282, (384, ), (1, ))
    assert_size_stride(primals_283, (196, 192), (192, 1))
    assert_size_stride(primals_284, (196, ), (1, ))
    assert_size_stride(primals_285, (384, ), (1, ))
    assert_size_stride(primals_286, (384, ), (1, ))
    assert_size_stride(primals_287, (1536, 384), (384, 1))
    assert_size_stride(primals_288, (1536, ), (1, ))
    assert_size_stride(primals_289, (384, 768), (768, 1))
    assert_size_stride(primals_290, (384, ), (1, ))
    assert_size_stride(primals_291, (384, ), (1, ))
    assert_size_stride(primals_292, (384, ), (1, ))
    assert_size_stride(primals_293, (1000, 384), (384, 1))
    assert_size_stride(primals_294, (1000, ), (1, ))
    assert_size_stride(primals_295, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((384, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_1
    del primals_295
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, primals_2, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_2
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf7 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del primals_4
    buf8 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_4], Original ATen: [aten.mm]
    extern_kernels.mm(buf7, reinterpret_tensor(primals_5, (196, 384), (1, 196), 0), out=buf8)
    buf9 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_2(c_void_p(buf8.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf9.data_ptr()))
    buf10 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, buf9, reinterpret_tensor(primals_7, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf10)
    del primals_8
    buf11 = buf3; del buf3  # reuse
    buf12 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf14 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf15 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_3(c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    del primals_10
    buf16 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_12, buf15, reinterpret_tensor(primals_11, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf16)
    del primals_12
    buf17 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_4(c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    buf18 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_14, buf17, reinterpret_tensor(primals_13, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf18)
    del primals_14
    buf19 = buf11; del buf11  # reuse
    buf20 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf22 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf23 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_5(c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    del primals_16
    buf24 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_18], Original ATen: [aten.mm]
    extern_kernels.mm(buf23, reinterpret_tensor(primals_17, (196, 384), (1, 196), 0), out=buf24)
    buf25 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_6(c_void_p(buf24.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf25.data_ptr()))
    buf26 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_20, buf25, reinterpret_tensor(primals_19, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf26)
    del primals_20
    buf27 = buf19; del buf19  # reuse
    buf28 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf30 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf31 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_7(c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del primals_22
    buf32 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_24, buf31, reinterpret_tensor(primals_23, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf32)
    del primals_24
    buf33 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_8(c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    buf34 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_26, buf33, reinterpret_tensor(primals_25, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf34)
    del primals_26
    buf35 = reinterpret_tensor(buf34, (8, 196, 384), (75264, 384, 1), 0); del buf34  # reuse
    buf36 = buf27; del buf27  # reuse
    buf37 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf39 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf40 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_9(c_void_p(buf35.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    del primals_28
    buf41 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_32], Original ATen: [aten.mm]
    extern_kernels.mm(buf40, reinterpret_tensor(primals_29, (196, 384), (1, 196), 0), out=buf41)
    buf42 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_10(c_void_p(buf41.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf42.data_ptr()))
    buf43 = buf26; del buf26  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_32, buf42, reinterpret_tensor(primals_31, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf43)
    del primals_32
    buf44 = buf36; del buf36  # reuse
    buf45 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf47 = reinterpret_tensor(buf2, (8, 196, 384), (75264, 384, 1), 0); del buf2  # reuse
    buf48 = buf18; del buf18  # reuse
    cpp_fused_add_native_layer_norm_view_11(c_void_p(buf35.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del primals_34
    buf49 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_36, buf48, reinterpret_tensor(primals_35, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf49)
    del primals_36
    buf50 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_12(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    buf51 = reinterpret_tensor(buf10, (1568, 384), (384, 1), 0); del buf10  # reuse
    # Source Nodes: [x_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_38, buf50, reinterpret_tensor(primals_37, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf51)
    del primals_38
    buf52 = buf44; del buf44  # reuse
    buf53 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf55 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf56 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_13(c_void_p(buf35.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    del primals_40
    buf57 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_46], Original ATen: [aten.mm]
    extern_kernels.mm(buf56, reinterpret_tensor(primals_41, (196, 384), (1, 196), 0), out=buf57)
    buf58 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_14(c_void_p(buf57.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf58.data_ptr()))
    buf59 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_44, buf58, reinterpret_tensor(primals_43, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf59)
    del primals_44
    buf60 = buf52; del buf52  # reuse
    buf61 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf63 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf64 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_15(c_void_p(buf35.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()))
    del primals_46
    buf65 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_48, buf64, reinterpret_tensor(primals_47, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf65)
    del primals_48
    buf66 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_16(c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    buf67 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_50, buf66, reinterpret_tensor(primals_49, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf67)
    del primals_50
    buf68 = reinterpret_tensor(buf67, (8, 196, 384), (75264, 384, 1), 0); del buf67  # reuse
    buf69 = buf60; del buf60  # reuse
    buf70 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf72 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf73 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_17(c_void_p(buf68.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_52
    buf74 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_60], Original ATen: [aten.mm]
    extern_kernels.mm(buf73, reinterpret_tensor(primals_53, (196, 384), (1, 196), 0), out=buf74)
    buf75 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_18(c_void_p(buf74.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = buf59; del buf59  # reuse
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_56, buf75, reinterpret_tensor(primals_55, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf76)
    del primals_56
    buf77 = buf69; del buf69  # reuse
    buf78 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf80 = reinterpret_tensor(buf51, (8, 196, 384), (75264, 384, 1), 0); del buf51  # reuse
    buf81 = reinterpret_tensor(buf43, (1568, 384), (384, 1), 0); del buf43  # reuse
    cpp_fused_add_native_layer_norm_view_19(c_void_p(buf68.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del primals_58
    buf82 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, buf81, reinterpret_tensor(primals_59, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf82)
    del primals_60
    buf83 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_20(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    buf84 = reinterpret_tensor(buf35, (1568, 384), (384, 1), 0); del buf35  # reuse
    # Source Nodes: [x_71], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_62, buf83, reinterpret_tensor(primals_61, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf84)
    del primals_62
    buf85 = buf77; del buf77  # reuse
    buf86 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf88 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf89 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_21(c_void_p(buf68.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    del primals_64
    buf90 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_74], Original ATen: [aten.mm]
    extern_kernels.mm(buf89, reinterpret_tensor(primals_65, (196, 384), (1, 196), 0), out=buf90)
    buf91 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_22(c_void_p(buf90.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf91.data_ptr()))
    buf92 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_68, buf91, reinterpret_tensor(primals_67, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf92)
    del primals_68
    buf93 = buf85; del buf85  # reuse
    buf94 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf96 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf97 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_23(c_void_p(buf68.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_70
    buf98 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_72, buf97, reinterpret_tensor(primals_71, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf98)
    del primals_72
    buf99 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_24(c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    buf100 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, buf99, reinterpret_tensor(primals_73, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf100)
    del primals_74
    buf101 = reinterpret_tensor(buf100, (8, 196, 384), (75264, 384, 1), 0); del buf100  # reuse
    buf102 = buf93; del buf93  # reuse
    buf103 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf105 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf106 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_25(c_void_p(buf101.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del primals_76
    buf107 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_88], Original ATen: [aten.mm]
    extern_kernels.mm(buf106, reinterpret_tensor(primals_77, (196, 384), (1, 196), 0), out=buf107)
    buf108 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_26(c_void_p(buf107.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf108.data_ptr()))
    buf109 = buf92; del buf92  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_80, buf108, reinterpret_tensor(primals_79, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf109)
    del primals_80
    buf110 = buf102; del buf102  # reuse
    buf111 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf113 = reinterpret_tensor(buf84, (8, 196, 384), (75264, 384, 1), 0); del buf84  # reuse
    buf114 = reinterpret_tensor(buf76, (1568, 384), (384, 1), 0); del buf76  # reuse
    cpp_fused_add_native_layer_norm_view_27(c_void_p(buf101.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()))
    del primals_82
    buf115 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_84, buf114, reinterpret_tensor(primals_83, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf115)
    del primals_84
    buf116 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_28(c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    buf117 = reinterpret_tensor(buf68, (1568, 384), (384, 1), 0); del buf68  # reuse
    # Source Nodes: [x_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, buf116, reinterpret_tensor(primals_85, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf117)
    del primals_86
    buf118 = buf110; del buf110  # reuse
    buf119 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf121 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf122 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_29(c_void_p(buf101.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    del primals_88
    buf123 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_102], Original ATen: [aten.mm]
    extern_kernels.mm(buf122, reinterpret_tensor(primals_89, (196, 384), (1, 196), 0), out=buf123)
    buf124 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_30(c_void_p(buf123.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf124.data_ptr()))
    buf125 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf124, reinterpret_tensor(primals_91, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf125)
    del primals_92
    buf126 = buf118; del buf118  # reuse
    buf127 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf129 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf130 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_31(c_void_p(buf101.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    del primals_94
    buf131 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_96, buf130, reinterpret_tensor(primals_95, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf131)
    del primals_96
    buf132 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_32(c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    buf133 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_98, buf132, reinterpret_tensor(primals_97, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf133)
    del primals_98
    buf134 = reinterpret_tensor(buf133, (8, 196, 384), (75264, 384, 1), 0); del buf133  # reuse
    buf135 = buf126; del buf126  # reuse
    buf136 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf138 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf139 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_33(c_void_p(buf134.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del primals_100
    buf140 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_116], Original ATen: [aten.mm]
    extern_kernels.mm(buf139, reinterpret_tensor(primals_101, (196, 384), (1, 196), 0), out=buf140)
    buf141 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_34(c_void_p(buf140.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf141.data_ptr()))
    buf142 = buf125; del buf125  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf141, reinterpret_tensor(primals_103, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf142)
    del primals_104
    buf143 = buf135; del buf135  # reuse
    buf144 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf146 = reinterpret_tensor(buf117, (8, 196, 384), (75264, 384, 1), 0); del buf117  # reuse
    buf147 = reinterpret_tensor(buf109, (1568, 384), (384, 1), 0); del buf109  # reuse
    cpp_fused_add_native_layer_norm_view_35(c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    del primals_106
    buf148 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_108, buf147, reinterpret_tensor(primals_107, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf148)
    del primals_108
    buf149 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_36(c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()))
    buf150 = reinterpret_tensor(buf101, (1568, 384), (384, 1), 0); del buf101  # reuse
    # Source Nodes: [x_127], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_110, buf149, reinterpret_tensor(primals_109, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf150)
    del primals_110
    buf151 = buf143; del buf143  # reuse
    buf152 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf154 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf155 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_37(c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    del primals_112
    buf156 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_130], Original ATen: [aten.mm]
    extern_kernels.mm(buf155, reinterpret_tensor(primals_113, (196, 384), (1, 196), 0), out=buf156)
    buf157 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_38(c_void_p(buf156.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf157.data_ptr()))
    buf158 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_116, buf157, reinterpret_tensor(primals_115, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf158)
    del primals_116
    buf159 = buf151; del buf151  # reuse
    buf160 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf162 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf163 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_39(c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    del primals_118
    buf164 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf163, reinterpret_tensor(primals_119, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf164)
    del primals_120
    buf165 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_40(c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    buf166 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_122, buf165, reinterpret_tensor(primals_121, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf166)
    del primals_122
    buf167 = reinterpret_tensor(buf166, (8, 196, 384), (75264, 384, 1), 0); del buf166  # reuse
    buf168 = buf159; del buf159  # reuse
    buf169 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf171 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf172 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_41(c_void_p(buf167.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del primals_124
    buf173 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_144], Original ATen: [aten.mm]
    extern_kernels.mm(buf172, reinterpret_tensor(primals_125, (196, 384), (1, 196), 0), out=buf173)
    buf174 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_42(c_void_p(buf173.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf174.data_ptr()))
    buf175 = buf158; del buf158  # reuse
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_128, buf174, reinterpret_tensor(primals_127, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf175)
    del primals_128
    buf176 = buf168; del buf168  # reuse
    buf177 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf179 = reinterpret_tensor(buf150, (8, 196, 384), (75264, 384, 1), 0); del buf150  # reuse
    buf180 = reinterpret_tensor(buf142, (1568, 384), (384, 1), 0); del buf142  # reuse
    cpp_fused_add_native_layer_norm_view_43(c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()))
    del primals_130
    buf181 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf180, reinterpret_tensor(primals_131, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf181)
    del primals_132
    buf182 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_44(c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()))
    buf183 = reinterpret_tensor(buf134, (1568, 384), (384, 1), 0); del buf134  # reuse
    # Source Nodes: [x_155], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_134, buf182, reinterpret_tensor(primals_133, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf183)
    del primals_134
    buf184 = buf176; del buf176  # reuse
    buf185 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf187 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf188 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_45(c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    del primals_136
    buf189 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_158], Original ATen: [aten.mm]
    extern_kernels.mm(buf188, reinterpret_tensor(primals_137, (196, 384), (1, 196), 0), out=buf189)
    buf190 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_46(c_void_p(buf189.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_140, buf190, reinterpret_tensor(primals_139, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf191)
    del primals_140
    buf192 = buf184; del buf184  # reuse
    buf193 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf195 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf196 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_47(c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()))
    del primals_142
    buf197 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf196, reinterpret_tensor(primals_143, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf197)
    del primals_144
    buf198 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_48(c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    buf199 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_169], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_146, buf198, reinterpret_tensor(primals_145, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf199)
    del primals_146
    buf200 = reinterpret_tensor(buf199, (8, 196, 384), (75264, 384, 1), 0); del buf199  # reuse
    buf201 = buf192; del buf192  # reuse
    buf202 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf204 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf205 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_49(c_void_p(buf200.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    del primals_148
    buf206 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_172], Original ATen: [aten.mm]
    extern_kernels.mm(buf205, reinterpret_tensor(primals_149, (196, 384), (1, 196), 0), out=buf206)
    buf207 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_50(c_void_p(buf206.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(buf207.data_ptr()))
    buf208 = buf191; del buf191  # reuse
    # Source Nodes: [x_176], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_152, buf207, reinterpret_tensor(primals_151, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf208)
    del primals_152
    buf209 = buf201; del buf201  # reuse
    buf210 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf212 = reinterpret_tensor(buf183, (8, 196, 384), (75264, 384, 1), 0); del buf183  # reuse
    buf213 = reinterpret_tensor(buf175, (1568, 384), (384, 1), 0); del buf175  # reuse
    cpp_fused_add_native_layer_norm_view_51(c_void_p(buf200.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del primals_154
    buf214 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_179], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_156, buf213, reinterpret_tensor(primals_155, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf214)
    del primals_156
    buf215 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_52(c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    buf216 = reinterpret_tensor(buf167, (1568, 384), (384, 1), 0); del buf167  # reuse
    # Source Nodes: [x_183], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_158, buf215, reinterpret_tensor(primals_157, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf216)
    del primals_158
    buf217 = buf209; del buf209  # reuse
    buf218 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf220 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf221 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_53(c_void_p(buf200.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    del primals_160
    buf222 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_186], Original ATen: [aten.mm]
    extern_kernels.mm(buf221, reinterpret_tensor(primals_161, (196, 384), (1, 196), 0), out=buf222)
    buf223 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_54(c_void_p(buf222.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(buf223.data_ptr()))
    buf224 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_190], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_164, buf223, reinterpret_tensor(primals_163, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf224)
    del primals_164
    buf225 = buf217; del buf217  # reuse
    buf226 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf228 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf229 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_55(c_void_p(buf200.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()))
    del primals_166
    buf230 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_193], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_168, buf229, reinterpret_tensor(primals_167, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf230)
    del primals_168
    buf231 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_56(c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    buf232 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_197], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_170, buf231, reinterpret_tensor(primals_169, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf232)
    del primals_170
    buf233 = reinterpret_tensor(buf232, (8, 196, 384), (75264, 384, 1), 0); del buf232  # reuse
    buf234 = buf225; del buf225  # reuse
    buf235 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf237 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf238 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_57(c_void_p(buf233.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del primals_172
    buf239 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_200], Original ATen: [aten.mm]
    extern_kernels.mm(buf238, reinterpret_tensor(primals_173, (196, 384), (1, 196), 0), out=buf239)
    buf240 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_58(c_void_p(buf239.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = buf224; del buf224  # reuse
    # Source Nodes: [x_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_176, buf240, reinterpret_tensor(primals_175, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf241)
    del primals_176
    buf242 = buf234; del buf234  # reuse
    buf243 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf245 = reinterpret_tensor(buf216, (8, 196, 384), (75264, 384, 1), 0); del buf216  # reuse
    buf246 = reinterpret_tensor(buf208, (1568, 384), (384, 1), 0); del buf208  # reuse
    cpp_fused_add_native_layer_norm_view_59(c_void_p(buf233.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()))
    del primals_178
    buf247 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_207], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_180, buf246, reinterpret_tensor(primals_179, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf247)
    del primals_180
    buf248 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_60(c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    buf249 = reinterpret_tensor(buf200, (1568, 384), (384, 1), 0); del buf200  # reuse
    # Source Nodes: [x_211], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_182, buf248, reinterpret_tensor(primals_181, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf249)
    del primals_182
    buf250 = buf242; del buf242  # reuse
    buf251 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf253 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf254 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_61(c_void_p(buf233.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    del primals_184
    buf255 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_214], Original ATen: [aten.mm]
    extern_kernels.mm(buf254, reinterpret_tensor(primals_185, (196, 384), (1, 196), 0), out=buf255)
    buf256 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_62(c_void_p(buf255.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(buf256.data_ptr()))
    buf257 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_218], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_188, buf256, reinterpret_tensor(primals_187, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf257)
    del primals_188
    buf258 = buf250; del buf250  # reuse
    buf259 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf261 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf262 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_63(c_void_p(buf233.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()))
    del primals_190
    buf263 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_221], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_192, buf262, reinterpret_tensor(primals_191, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf263)
    del primals_192
    buf264 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_64(c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    buf265 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_225], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_194, buf264, reinterpret_tensor(primals_193, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf265)
    del primals_194
    buf266 = reinterpret_tensor(buf265, (8, 196, 384), (75264, 384, 1), 0); del buf265  # reuse
    buf267 = buf258; del buf258  # reuse
    buf268 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf270 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf271 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_65(c_void_p(buf266.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    del primals_196
    buf272 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_228], Original ATen: [aten.mm]
    extern_kernels.mm(buf271, reinterpret_tensor(primals_197, (196, 384), (1, 196), 0), out=buf272)
    buf273 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_66(c_void_p(buf272.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(buf273.data_ptr()))
    buf274 = buf257; del buf257  # reuse
    # Source Nodes: [x_232], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_200, buf273, reinterpret_tensor(primals_199, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf274)
    del primals_200
    buf275 = buf267; del buf267  # reuse
    buf276 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf278 = reinterpret_tensor(buf249, (8, 196, 384), (75264, 384, 1), 0); del buf249  # reuse
    buf279 = reinterpret_tensor(buf241, (1568, 384), (384, 1), 0); del buf241  # reuse
    cpp_fused_add_native_layer_norm_view_67(c_void_p(buf266.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    del primals_202
    buf280 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_235], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_204, buf279, reinterpret_tensor(primals_203, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf280)
    del primals_204
    buf281 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_68(c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    buf282 = reinterpret_tensor(buf233, (1568, 384), (384, 1), 0); del buf233  # reuse
    # Source Nodes: [x_239], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_206, buf281, reinterpret_tensor(primals_205, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf282)
    del primals_206
    buf283 = buf275; del buf275  # reuse
    buf284 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf286 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf287 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_69(c_void_p(buf266.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    del primals_208
    buf288 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_242], Original ATen: [aten.mm]
    extern_kernels.mm(buf287, reinterpret_tensor(primals_209, (196, 384), (1, 196), 0), out=buf288)
    buf289 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_70(c_void_p(buf288.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_246], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_212, buf289, reinterpret_tensor(primals_211, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf290)
    del primals_212
    buf291 = buf283; del buf283  # reuse
    buf292 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf294 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf295 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_71(c_void_p(buf266.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    del primals_214
    buf296 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_249], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_216, buf295, reinterpret_tensor(primals_215, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf296)
    del primals_216
    buf297 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_72(c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    buf298 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_253], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_218, buf297, reinterpret_tensor(primals_217, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf298)
    del primals_218
    buf299 = reinterpret_tensor(buf298, (8, 196, 384), (75264, 384, 1), 0); del buf298  # reuse
    buf300 = buf291; del buf291  # reuse
    buf301 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf303 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf304 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_73(c_void_p(buf299.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    del primals_220
    buf305 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_256], Original ATen: [aten.mm]
    extern_kernels.mm(buf304, reinterpret_tensor(primals_221, (196, 384), (1, 196), 0), out=buf305)
    buf306 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_74(c_void_p(buf305.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(buf306.data_ptr()))
    buf307 = buf290; del buf290  # reuse
    # Source Nodes: [x_260], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_224, buf306, reinterpret_tensor(primals_223, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf307)
    del primals_224
    buf308 = buf300; del buf300  # reuse
    buf309 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf311 = reinterpret_tensor(buf282, (8, 196, 384), (75264, 384, 1), 0); del buf282  # reuse
    buf312 = reinterpret_tensor(buf274, (1568, 384), (384, 1), 0); del buf274  # reuse
    cpp_fused_add_native_layer_norm_view_75(c_void_p(buf299.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del primals_226
    buf313 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_263], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_228, buf312, reinterpret_tensor(primals_227, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf313)
    del primals_228
    buf314 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_76(c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    buf315 = reinterpret_tensor(buf266, (1568, 384), (384, 1), 0); del buf266  # reuse
    # Source Nodes: [x_267], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_230, buf314, reinterpret_tensor(primals_229, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf315)
    del primals_230
    buf316 = buf308; del buf308  # reuse
    buf317 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf319 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf320 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_77(c_void_p(buf299.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    del primals_232
    buf321 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_270], Original ATen: [aten.mm]
    extern_kernels.mm(buf320, reinterpret_tensor(primals_233, (196, 384), (1, 196), 0), out=buf321)
    buf322 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_78(c_void_p(buf321.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(buf322.data_ptr()))
    buf323 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_274], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_236, buf322, reinterpret_tensor(primals_235, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf323)
    del primals_236
    buf324 = buf316; del buf316  # reuse
    buf325 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf327 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf328 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_79(c_void_p(buf299.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    del primals_238
    buf329 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_277], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_240, buf328, reinterpret_tensor(primals_239, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf329)
    del primals_240
    buf330 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_80(c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()))
    buf331 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_281], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_242, buf330, reinterpret_tensor(primals_241, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf331)
    del primals_242
    buf332 = reinterpret_tensor(buf331, (8, 196, 384), (75264, 384, 1), 0); del buf331  # reuse
    buf333 = buf324; del buf324  # reuse
    buf334 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf336 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf337 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_81(c_void_p(buf332.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    del primals_244
    buf338 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_284], Original ATen: [aten.mm]
    extern_kernels.mm(buf337, reinterpret_tensor(primals_245, (196, 384), (1, 196), 0), out=buf338)
    buf339 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_82(c_void_p(buf338.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(buf339.data_ptr()))
    buf340 = buf323; del buf323  # reuse
    # Source Nodes: [x_288], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_248, buf339, reinterpret_tensor(primals_247, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf340)
    del primals_248
    buf341 = buf333; del buf333  # reuse
    buf342 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf344 = reinterpret_tensor(buf315, (8, 196, 384), (75264, 384, 1), 0); del buf315  # reuse
    buf345 = reinterpret_tensor(buf307, (1568, 384), (384, 1), 0); del buf307  # reuse
    cpp_fused_add_native_layer_norm_view_83(c_void_p(buf332.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()))
    del primals_250
    buf346 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_291], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_252, buf345, reinterpret_tensor(primals_251, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf346)
    del primals_252
    buf347 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_84(c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()))
    buf348 = reinterpret_tensor(buf299, (1568, 384), (384, 1), 0); del buf299  # reuse
    # Source Nodes: [x_295], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_254, buf347, reinterpret_tensor(primals_253, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf348)
    del primals_254
    buf349 = buf341; del buf341  # reuse
    buf350 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf352 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf353 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_85(c_void_p(buf332.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    del primals_256
    buf354 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_298], Original ATen: [aten.mm]
    extern_kernels.mm(buf353, reinterpret_tensor(primals_257, (196, 384), (1, 196), 0), out=buf354)
    buf355 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_86(c_void_p(buf354.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(buf355.data_ptr()))
    buf356 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_302], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_260, buf355, reinterpret_tensor(primals_259, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf356)
    del primals_260
    buf357 = buf349; del buf349  # reuse
    buf358 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf360 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf361 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_87(c_void_p(buf332.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()))
    del primals_262
    buf362 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_305], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_264, buf361, reinterpret_tensor(primals_263, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf362)
    del primals_264
    buf363 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_88(c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    buf364 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_309], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_266, buf363, reinterpret_tensor(primals_265, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf364)
    del primals_266
    buf365 = reinterpret_tensor(buf364, (8, 196, 384), (75264, 384, 1), 0); del buf364  # reuse
    buf366 = buf357; del buf357  # reuse
    buf367 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf369 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf370 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_89(c_void_p(buf365.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()))
    del primals_268
    buf371 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_312], Original ATen: [aten.mm]
    extern_kernels.mm(buf370, reinterpret_tensor(primals_269, (196, 384), (1, 196), 0), out=buf371)
    buf372 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_90(c_void_p(buf371.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(buf372.data_ptr()))
    buf373 = buf356; del buf356  # reuse
    # Source Nodes: [x_316], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_272, buf372, reinterpret_tensor(primals_271, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf373)
    del primals_272
    buf374 = buf366; del buf366  # reuse
    buf375 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf377 = reinterpret_tensor(buf348, (8, 196, 384), (75264, 384, 1), 0); del buf348  # reuse
    buf378 = reinterpret_tensor(buf340, (1568, 384), (384, 1), 0); del buf340  # reuse
    cpp_fused_add_native_layer_norm_view_91(c_void_p(buf365.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()))
    del primals_274
    buf379 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_319], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_276, buf378, reinterpret_tensor(primals_275, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf379)
    del primals_276
    buf380 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_92(c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()))
    buf381 = reinterpret_tensor(buf332, (1568, 384), (384, 1), 0); del buf332  # reuse
    # Source Nodes: [x_323], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_278, buf380, reinterpret_tensor(primals_277, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf381)
    del primals_278
    buf382 = buf374; del buf374  # reuse
    buf383 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf385 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf386 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_93(c_void_p(buf365.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    del primals_280
    buf387 = empty((3072, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_326], Original ATen: [aten.mm]
    extern_kernels.mm(buf386, reinterpret_tensor(primals_281, (196, 384), (1, 196), 0), out=buf387)
    buf388 = empty((3072, 192), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_94(c_void_p(buf387.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(buf388.data_ptr()))
    buf389 = empty((3072, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_330], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_284, buf388, reinterpret_tensor(primals_283, (192, 196), (1, 192), 0), alpha=1, beta=1, out=buf389)
    del primals_284
    buf390 = buf382; del buf382  # reuse
    buf391 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf393 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf394 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_95(c_void_p(buf365.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    del primals_286
    buf395 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_333], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_288, buf394, reinterpret_tensor(primals_287, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf395)
    del primals_288
    buf396 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_mul_silu_view_96(c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    buf397 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_337], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_290, buf396, reinterpret_tensor(primals_289, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf397)
    del primals_290
    buf398 = reinterpret_tensor(buf397, (8, 196, 384), (75264, 384, 1), 0); del buf397  # reuse
    buf399 = buf390; del buf390  # reuse
    buf400 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf402 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf403 = empty((8, 384), device='cpu', dtype=torch.float32)
    buf404 = buf403; del buf403  # reuse
    cpp_fused_add_mean_native_layer_norm_97(c_void_p(buf398.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf402.data_ptr()))
    del buf365
    del buf373
    del buf381
    del buf389
    del buf398
    del buf399
    del primals_292
    buf405 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_294, buf404, reinterpret_tensor(primals_293, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf405)
    del primals_294
    buf406 = reinterpret_tensor(buf400, (8, 196, 1), (196, 1, 1), 0); del buf400  # reuse
    buf407 = reinterpret_tensor(buf391, (8, 196, 1), (196, 1, 1), 0); del buf391  # reuse
    buf408 = reinterpret_tensor(buf383, (8, 196, 1), (196, 1, 1), 0); del buf383  # reuse
    buf409 = reinterpret_tensor(buf375, (8, 196, 1), (196, 1, 1), 0); del buf375  # reuse
    buf410 = reinterpret_tensor(buf367, (8, 196, 1), (196, 1, 1), 0); del buf367  # reuse
    buf411 = reinterpret_tensor(buf358, (8, 196, 1), (196, 1, 1), 0); del buf358  # reuse
    buf412 = reinterpret_tensor(buf350, (8, 196, 1), (196, 1, 1), 0); del buf350  # reuse
    buf413 = reinterpret_tensor(buf342, (8, 196, 1), (196, 1, 1), 0); del buf342  # reuse
    buf414 = reinterpret_tensor(buf334, (8, 196, 1), (196, 1, 1), 0); del buf334  # reuse
    buf415 = reinterpret_tensor(buf325, (8, 196, 1), (196, 1, 1), 0); del buf325  # reuse
    buf416 = reinterpret_tensor(buf317, (8, 196, 1), (196, 1, 1), 0); del buf317  # reuse
    buf417 = reinterpret_tensor(buf309, (8, 196, 1), (196, 1, 1), 0); del buf309  # reuse
    buf418 = reinterpret_tensor(buf301, (8, 196, 1), (196, 1, 1), 0); del buf301  # reuse
    buf419 = reinterpret_tensor(buf292, (8, 196, 1), (196, 1, 1), 0); del buf292  # reuse
    buf420 = reinterpret_tensor(buf284, (8, 196, 1), (196, 1, 1), 0); del buf284  # reuse
    buf421 = reinterpret_tensor(buf276, (8, 196, 1), (196, 1, 1), 0); del buf276  # reuse
    buf422 = reinterpret_tensor(buf268, (8, 196, 1), (196, 1, 1), 0); del buf268  # reuse
    buf423 = reinterpret_tensor(buf259, (8, 196, 1), (196, 1, 1), 0); del buf259  # reuse
    buf424 = reinterpret_tensor(buf251, (8, 196, 1), (196, 1, 1), 0); del buf251  # reuse
    buf425 = reinterpret_tensor(buf243, (8, 196, 1), (196, 1, 1), 0); del buf243  # reuse
    buf426 = reinterpret_tensor(buf235, (8, 196, 1), (196, 1, 1), 0); del buf235  # reuse
    buf427 = reinterpret_tensor(buf226, (8, 196, 1), (196, 1, 1), 0); del buf226  # reuse
    buf428 = reinterpret_tensor(buf218, (8, 196, 1), (196, 1, 1), 0); del buf218  # reuse
    buf429 = reinterpret_tensor(buf210, (8, 196, 1), (196, 1, 1), 0); del buf210  # reuse
    buf430 = reinterpret_tensor(buf202, (8, 196, 1), (196, 1, 1), 0); del buf202  # reuse
    buf431 = reinterpret_tensor(buf193, (8, 196, 1), (196, 1, 1), 0); del buf193  # reuse
    buf432 = reinterpret_tensor(buf185, (8, 196, 1), (196, 1, 1), 0); del buf185  # reuse
    buf433 = reinterpret_tensor(buf177, (8, 196, 1), (196, 1, 1), 0); del buf177  # reuse
    buf434 = reinterpret_tensor(buf169, (8, 196, 1), (196, 1, 1), 0); del buf169  # reuse
    buf435 = reinterpret_tensor(buf160, (8, 196, 1), (196, 1, 1), 0); del buf160  # reuse
    buf436 = reinterpret_tensor(buf152, (8, 196, 1), (196, 1, 1), 0); del buf152  # reuse
    buf437 = reinterpret_tensor(buf144, (8, 196, 1), (196, 1, 1), 0); del buf144  # reuse
    buf438 = reinterpret_tensor(buf136, (8, 196, 1), (196, 1, 1), 0); del buf136  # reuse
    buf439 = reinterpret_tensor(buf127, (8, 196, 1), (196, 1, 1), 0); del buf127  # reuse
    buf440 = reinterpret_tensor(buf119, (8, 196, 1), (196, 1, 1), 0); del buf119  # reuse
    buf441 = reinterpret_tensor(buf111, (8, 196, 1), (196, 1, 1), 0); del buf111  # reuse
    buf442 = reinterpret_tensor(buf103, (8, 196, 1), (196, 1, 1), 0); del buf103  # reuse
    buf443 = reinterpret_tensor(buf94, (8, 196, 1), (196, 1, 1), 0); del buf94  # reuse
    buf444 = reinterpret_tensor(buf86, (8, 196, 1), (196, 1, 1), 0); del buf86  # reuse
    buf445 = reinterpret_tensor(buf78, (8, 196, 1), (196, 1, 1), 0); del buf78  # reuse
    buf446 = reinterpret_tensor(buf70, (8, 196, 1), (196, 1, 1), 0); del buf70  # reuse
    buf447 = reinterpret_tensor(buf61, (8, 196, 1), (196, 1, 1), 0); del buf61  # reuse
    buf448 = reinterpret_tensor(buf53, (8, 196, 1), (196, 1, 1), 0); del buf53  # reuse
    buf449 = reinterpret_tensor(buf45, (8, 196, 1), (196, 1, 1), 0); del buf45  # reuse
    buf450 = reinterpret_tensor(buf37, (8, 196, 1), (196, 1, 1), 0); del buf37  # reuse
    buf451 = reinterpret_tensor(buf28, (8, 196, 1), (196, 1, 1), 0); del buf28  # reuse
    buf452 = reinterpret_tensor(buf20, (8, 196, 1), (196, 1, 1), 0); del buf20  # reuse
    buf453 = reinterpret_tensor(buf12, (8, 196, 1), (196, 1, 1), 0); del buf12  # reuse
    buf454 = reinterpret_tensor(buf4, (8, 196, 1), (196, 1, 1), 0); del buf4  # reuse
    buf455 = reinterpret_tensor(buf8, (8, 384, 384), (147456, 384, 1), 0); del buf8  # reuse
    buf456 = reinterpret_tensor(buf24, (8, 384, 384), (147456, 384, 1), 0); del buf24  # reuse
    buf457 = reinterpret_tensor(buf41, (8, 384, 384), (147456, 384, 1), 0); del buf41  # reuse
    buf458 = reinterpret_tensor(buf57, (8, 384, 384), (147456, 384, 1), 0); del buf57  # reuse
    buf459 = reinterpret_tensor(buf74, (8, 384, 384), (147456, 384, 1), 0); del buf74  # reuse
    buf460 = reinterpret_tensor(buf90, (8, 384, 384), (147456, 384, 1), 0); del buf90  # reuse
    buf461 = reinterpret_tensor(buf107, (8, 384, 384), (147456, 384, 1), 0); del buf107  # reuse
    buf462 = reinterpret_tensor(buf123, (8, 384, 384), (147456, 384, 1), 0); del buf123  # reuse
    buf463 = reinterpret_tensor(buf140, (8, 384, 384), (147456, 384, 1), 0); del buf140  # reuse
    buf464 = reinterpret_tensor(buf156, (8, 384, 384), (147456, 384, 1), 0); del buf156  # reuse
    buf465 = reinterpret_tensor(buf173, (8, 384, 384), (147456, 384, 1), 0); del buf173  # reuse
    buf466 = reinterpret_tensor(buf189, (8, 384, 384), (147456, 384, 1), 0); del buf189  # reuse
    buf467 = reinterpret_tensor(buf206, (8, 384, 384), (147456, 384, 1), 0); del buf206  # reuse
    buf468 = reinterpret_tensor(buf222, (8, 384, 384), (147456, 384, 1), 0); del buf222  # reuse
    buf469 = reinterpret_tensor(buf239, (8, 384, 384), (147456, 384, 1), 0); del buf239  # reuse
    buf470 = reinterpret_tensor(buf255, (8, 384, 384), (147456, 384, 1), 0); del buf255  # reuse
    buf471 = reinterpret_tensor(buf272, (8, 384, 384), (147456, 384, 1), 0); del buf272  # reuse
    buf472 = reinterpret_tensor(buf288, (8, 384, 384), (147456, 384, 1), 0); del buf288  # reuse
    buf473 = reinterpret_tensor(buf305, (8, 384, 384), (147456, 384, 1), 0); del buf305  # reuse
    buf474 = reinterpret_tensor(buf321, (8, 384, 384), (147456, 384, 1), 0); del buf321  # reuse
    buf475 = reinterpret_tensor(buf338, (8, 384, 384), (147456, 384, 1), 0); del buf338  # reuse
    buf476 = reinterpret_tensor(buf354, (8, 384, 384), (147456, 384, 1), 0); del buf354  # reuse
    buf477 = reinterpret_tensor(buf371, (8, 384, 384), (147456, 384, 1), 0); del buf371  # reuse
    buf478 = reinterpret_tensor(buf387, (8, 384, 384), (147456, 384, 1), 0); del buf387  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_98(c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_282.data_ptr()))
    del primals_102
    del primals_114
    del primals_126
    del primals_138
    del primals_150
    del primals_162
    del primals_174
    del primals_18
    del primals_186
    del primals_198
    del primals_210
    del primals_222
    del primals_234
    del primals_246
    del primals_258
    del primals_270
    del primals_282
    del primals_30
    del primals_42
    del primals_54
    del primals_6
    del primals_66
    del primals_78
    del primals_90
    return (buf405, buf0, primals_3, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_159, primals_165, primals_171, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_231, primals_237, primals_243, primals_249, primals_255, primals_261, primals_267, primals_273, primals_279, primals_285, primals_291, buf1, buf6, buf7, reinterpret_tensor(buf455, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf455, (8, 384, 192), (147456, 384, 1), 192), buf9, buf14, buf15, reinterpret_tensor(buf16, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf16, (8, 196, 768), (301056, 1536, 1), 768), buf17, buf22, buf23, reinterpret_tensor(buf456, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf456, (8, 384, 192), (147456, 384, 1), 192), buf25, buf30, buf31, reinterpret_tensor(buf32, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf32, (8, 196, 768), (301056, 1536, 1), 768), buf33, buf39, buf40, reinterpret_tensor(buf457, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf457, (8, 384, 192), (147456, 384, 1), 192), buf42, buf47, buf48, reinterpret_tensor(buf49, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf49, (8, 196, 768), (301056, 1536, 1), 768), buf50, buf55, buf56, reinterpret_tensor(buf458, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf458, (8, 384, 192), (147456, 384, 1), 192), buf58, buf63, buf64, reinterpret_tensor(buf65, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf65, (8, 196, 768), (301056, 1536, 1), 768), buf66, buf72, buf73, reinterpret_tensor(buf459, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf459, (8, 384, 192), (147456, 384, 1), 192), buf75, buf80, buf81, reinterpret_tensor(buf82, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf82, (8, 196, 768), (301056, 1536, 1), 768), buf83, buf88, buf89, reinterpret_tensor(buf460, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf460, (8, 384, 192), (147456, 384, 1), 192), buf91, buf96, buf97, reinterpret_tensor(buf98, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf98, (8, 196, 768), (301056, 1536, 1), 768), buf99, buf105, buf106, reinterpret_tensor(buf461, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf461, (8, 384, 192), (147456, 384, 1), 192), buf108, buf113, buf114, reinterpret_tensor(buf115, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf115, (8, 196, 768), (301056, 1536, 1), 768), buf116, buf121, buf122, reinterpret_tensor(buf462, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf462, (8, 384, 192), (147456, 384, 1), 192), buf124, buf129, buf130, reinterpret_tensor(buf131, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf131, (8, 196, 768), (301056, 1536, 1), 768), buf132, buf138, buf139, reinterpret_tensor(buf463, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf463, (8, 384, 192), (147456, 384, 1), 192), buf141, buf146, buf147, reinterpret_tensor(buf148, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf148, (8, 196, 768), (301056, 1536, 1), 768), buf149, buf154, buf155, reinterpret_tensor(buf464, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf464, (8, 384, 192), (147456, 384, 1), 192), buf157, buf162, buf163, reinterpret_tensor(buf164, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf164, (8, 196, 768), (301056, 1536, 1), 768), buf165, buf171, buf172, reinterpret_tensor(buf465, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf465, (8, 384, 192), (147456, 384, 1), 192), buf174, buf179, buf180, reinterpret_tensor(buf181, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf181, (8, 196, 768), (301056, 1536, 1), 768), buf182, buf187, buf188, reinterpret_tensor(buf466, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf466, (8, 384, 192), (147456, 384, 1), 192), buf190, buf195, buf196, reinterpret_tensor(buf197, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf197, (8, 196, 768), (301056, 1536, 1), 768), buf198, buf204, buf205, reinterpret_tensor(buf467, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf467, (8, 384, 192), (147456, 384, 1), 192), buf207, buf212, buf213, reinterpret_tensor(buf214, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf214, (8, 196, 768), (301056, 1536, 1), 768), buf215, buf220, buf221, reinterpret_tensor(buf468, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf468, (8, 384, 192), (147456, 384, 1), 192), buf223, buf228, buf229, reinterpret_tensor(buf230, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf230, (8, 196, 768), (301056, 1536, 1), 768), buf231, buf237, buf238, reinterpret_tensor(buf469, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf469, (8, 384, 192), (147456, 384, 1), 192), buf240, buf245, buf246, reinterpret_tensor(buf247, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf247, (8, 196, 768), (301056, 1536, 1), 768), buf248, buf253, buf254, reinterpret_tensor(buf470, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf470, (8, 384, 192), (147456, 384, 1), 192), buf256, buf261, buf262, reinterpret_tensor(buf263, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf263, (8, 196, 768), (301056, 1536, 1), 768), buf264, buf270, buf271, reinterpret_tensor(buf471, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf471, (8, 384, 192), (147456, 384, 1), 192), buf273, buf278, buf279, reinterpret_tensor(buf280, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf280, (8, 196, 768), (301056, 1536, 1), 768), buf281, buf286, buf287, reinterpret_tensor(buf472, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf472, (8, 384, 192), (147456, 384, 1), 192), buf289, buf294, buf295, reinterpret_tensor(buf296, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf296, (8, 196, 768), (301056, 1536, 1), 768), buf297, buf303, buf304, reinterpret_tensor(buf473, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf473, (8, 384, 192), (147456, 384, 1), 192), buf306, buf311, buf312, reinterpret_tensor(buf313, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf313, (8, 196, 768), (301056, 1536, 1), 768), buf314, buf319, buf320, reinterpret_tensor(buf474, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf474, (8, 384, 192), (147456, 384, 1), 192), buf322, buf327, buf328, reinterpret_tensor(buf329, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf329, (8, 196, 768), (301056, 1536, 1), 768), buf330, buf336, buf337, reinterpret_tensor(buf475, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf475, (8, 384, 192), (147456, 384, 1), 192), buf339, buf344, buf345, reinterpret_tensor(buf346, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf346, (8, 196, 768), (301056, 1536, 1), 768), buf347, buf352, buf353, reinterpret_tensor(buf476, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf476, (8, 384, 192), (147456, 384, 1), 192), buf355, buf360, buf361, reinterpret_tensor(buf362, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf362, (8, 196, 768), (301056, 1536, 1), 768), buf363, buf369, buf370, reinterpret_tensor(buf477, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf477, (8, 384, 192), (147456, 384, 1), 192), buf372, buf377, buf378, reinterpret_tensor(buf379, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf379, (8, 196, 768), (301056, 1536, 1), 768), buf380, buf385, buf386, reinterpret_tensor(buf478, (8, 384, 192), (147456, 384, 1), 0), reinterpret_tensor(buf478, (8, 384, 192), (147456, 384, 1), 192), buf388, buf393, buf394, reinterpret_tensor(buf395, (8, 196, 768), (301056, 1536, 1), 0), reinterpret_tensor(buf395, (8, 196, 768), (301056, 1536, 1), 768), buf396, buf402, buf404, reinterpret_tensor(primals_293, (1000, 384), (384, 1), 0), buf406, reinterpret_tensor(primals_289, (384, 768), (768, 1), 0), reinterpret_tensor(primals_287, (1536, 384), (384, 1), 0), buf407, reinterpret_tensor(primals_283, (196, 192), (192, 1), 0), reinterpret_tensor(primals_281, (384, 196), (196, 1), 0), buf408, reinterpret_tensor(primals_277, (384, 768), (768, 1), 0), reinterpret_tensor(primals_275, (1536, 384), (384, 1), 0), buf409, reinterpret_tensor(primals_271, (196, 192), (192, 1), 0), reinterpret_tensor(primals_269, (384, 196), (196, 1), 0), buf410, reinterpret_tensor(primals_265, (384, 768), (768, 1), 0), reinterpret_tensor(primals_263, (1536, 384), (384, 1), 0), buf411, reinterpret_tensor(primals_259, (196, 192), (192, 1), 0), reinterpret_tensor(primals_257, (384, 196), (196, 1), 0), buf412, reinterpret_tensor(primals_253, (384, 768), (768, 1), 0), reinterpret_tensor(primals_251, (1536, 384), (384, 1), 0), buf413, reinterpret_tensor(primals_247, (196, 192), (192, 1), 0), reinterpret_tensor(primals_245, (384, 196), (196, 1), 0), buf414, reinterpret_tensor(primals_241, (384, 768), (768, 1), 0), reinterpret_tensor(primals_239, (1536, 384), (384, 1), 0), buf415, reinterpret_tensor(primals_235, (196, 192), (192, 1), 0), reinterpret_tensor(primals_233, (384, 196), (196, 1), 0), buf416, reinterpret_tensor(primals_229, (384, 768), (768, 1), 0), reinterpret_tensor(primals_227, (1536, 384), (384, 1), 0), buf417, reinterpret_tensor(primals_223, (196, 192), (192, 1), 0), reinterpret_tensor(primals_221, (384, 196), (196, 1), 0), buf418, reinterpret_tensor(primals_217, (384, 768), (768, 1), 0), reinterpret_tensor(primals_215, (1536, 384), (384, 1), 0), buf419, reinterpret_tensor(primals_211, (196, 192), (192, 1), 0), reinterpret_tensor(primals_209, (384, 196), (196, 1), 0), buf420, reinterpret_tensor(primals_205, (384, 768), (768, 1), 0), reinterpret_tensor(primals_203, (1536, 384), (384, 1), 0), buf421, reinterpret_tensor(primals_199, (196, 192), (192, 1), 0), reinterpret_tensor(primals_197, (384, 196), (196, 1), 0), buf422, reinterpret_tensor(primals_193, (384, 768), (768, 1), 0), reinterpret_tensor(primals_191, (1536, 384), (384, 1), 0), buf423, reinterpret_tensor(primals_187, (196, 192), (192, 1), 0), reinterpret_tensor(primals_185, (384, 196), (196, 1), 0), buf424, reinterpret_tensor(primals_181, (384, 768), (768, 1), 0), reinterpret_tensor(primals_179, (1536, 384), (384, 1), 0), buf425, reinterpret_tensor(primals_175, (196, 192), (192, 1), 0), reinterpret_tensor(primals_173, (384, 196), (196, 1), 0), buf426, reinterpret_tensor(primals_169, (384, 768), (768, 1), 0), reinterpret_tensor(primals_167, (1536, 384), (384, 1), 0), buf427, reinterpret_tensor(primals_163, (196, 192), (192, 1), 0), reinterpret_tensor(primals_161, (384, 196), (196, 1), 0), buf428, reinterpret_tensor(primals_157, (384, 768), (768, 1), 0), reinterpret_tensor(primals_155, (1536, 384), (384, 1), 0), buf429, reinterpret_tensor(primals_151, (196, 192), (192, 1), 0), reinterpret_tensor(primals_149, (384, 196), (196, 1), 0), buf430, reinterpret_tensor(primals_145, (384, 768), (768, 1), 0), reinterpret_tensor(primals_143, (1536, 384), (384, 1), 0), buf431, reinterpret_tensor(primals_139, (196, 192), (192, 1), 0), reinterpret_tensor(primals_137, (384, 196), (196, 1), 0), buf432, reinterpret_tensor(primals_133, (384, 768), (768, 1), 0), reinterpret_tensor(primals_131, (1536, 384), (384, 1), 0), buf433, reinterpret_tensor(primals_127, (196, 192), (192, 1), 0), reinterpret_tensor(primals_125, (384, 196), (196, 1), 0), buf434, reinterpret_tensor(primals_121, (384, 768), (768, 1), 0), reinterpret_tensor(primals_119, (1536, 384), (384, 1), 0), buf435, reinterpret_tensor(primals_115, (196, 192), (192, 1), 0), reinterpret_tensor(primals_113, (384, 196), (196, 1), 0), buf436, reinterpret_tensor(primals_109, (384, 768), (768, 1), 0), reinterpret_tensor(primals_107, (1536, 384), (384, 1), 0), buf437, reinterpret_tensor(primals_103, (196, 192), (192, 1), 0), reinterpret_tensor(primals_101, (384, 196), (196, 1), 0), buf438, reinterpret_tensor(primals_97, (384, 768), (768, 1), 0), reinterpret_tensor(primals_95, (1536, 384), (384, 1), 0), buf439, reinterpret_tensor(primals_91, (196, 192), (192, 1), 0), reinterpret_tensor(primals_89, (384, 196), (196, 1), 0), buf440, reinterpret_tensor(primals_85, (384, 768), (768, 1), 0), reinterpret_tensor(primals_83, (1536, 384), (384, 1), 0), buf441, reinterpret_tensor(primals_79, (196, 192), (192, 1), 0), reinterpret_tensor(primals_77, (384, 196), (196, 1), 0), buf442, reinterpret_tensor(primals_73, (384, 768), (768, 1), 0), reinterpret_tensor(primals_71, (1536, 384), (384, 1), 0), buf443, reinterpret_tensor(primals_67, (196, 192), (192, 1), 0), reinterpret_tensor(primals_65, (384, 196), (196, 1), 0), buf444, reinterpret_tensor(primals_61, (384, 768), (768, 1), 0), reinterpret_tensor(primals_59, (1536, 384), (384, 1), 0), buf445, reinterpret_tensor(primals_55, (196, 192), (192, 1), 0), reinterpret_tensor(primals_53, (384, 196), (196, 1), 0), buf446, reinterpret_tensor(primals_49, (384, 768), (768, 1), 0), reinterpret_tensor(primals_47, (1536, 384), (384, 1), 0), buf447, reinterpret_tensor(primals_43, (196, 192), (192, 1), 0), reinterpret_tensor(primals_41, (384, 196), (196, 1), 0), buf448, reinterpret_tensor(primals_37, (384, 768), (768, 1), 0), reinterpret_tensor(primals_35, (1536, 384), (384, 1), 0), buf449, reinterpret_tensor(primals_31, (196, 192), (192, 1), 0), reinterpret_tensor(primals_29, (384, 196), (196, 1), 0), buf450, reinterpret_tensor(primals_25, (384, 768), (768, 1), 0), reinterpret_tensor(primals_23, (1536, 384), (384, 1), 0), buf451, reinterpret_tensor(primals_19, (196, 192), (192, 1), 0), reinterpret_tensor(primals_17, (384, 196), (196, 1), 0), buf452, reinterpret_tensor(primals_13, (384, 768), (768, 1), 0), reinterpret_tensor(primals_11, (1536, 384), (384, 1), 0), buf453, reinterpret_tensor(primals_7, (196, 192), (192, 1), 0), reinterpret_tensor(primals_5, (384, 196), (196, 1), 0), buf454, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmixer_24_224', benchmark_compiled_module)
