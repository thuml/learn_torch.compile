
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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


cpp_fused_native_layer_norm_1 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_2 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_5 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_8 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_11 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_14 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_17 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
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
                       const float* in_ptr4,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_20 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_23 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_26 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_29 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_32 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_35 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_38 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_41 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
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
                       const float* in_ptr4,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_44 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_47 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_50 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_53 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_56 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_59 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_62 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_65 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
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
                       const float* in_ptr4,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_68 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_71 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_74 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_77 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(256.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_80 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(256.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_83 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_86 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(256.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_89 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x0)));
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
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x0)));
                            auto tmp24 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp27 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp2 = static_cast<float>(0.5);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(0.7071067811865476);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp1 * tmp6;
                            auto tmp8 = tmp7.erf();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp16 = static_cast<float>(768.0);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp18 + tmp20;
                            auto tmp22 = tmp21.rsqrt();
                            auto tmp23 = tmp14 * tmp22;
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp28 = at::vec::Vectorized<float>(tmp27);
                            auto tmp29 = tmp26 + tmp28;
                            tmp29.store(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp12 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                        auto tmp15 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
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
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = static_cast<float>(768.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp26 = tmp24 + tmp25;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp26.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                        float tmp12[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L), tmp12, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(tmp12 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 + tmp15;
                            auto tmp17 = tmp11 * tmp16;
                            tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp9 = in_ptr1[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                        auto tmp10 = in_ptr2[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp3 = static_cast<float>(0.7071067811865476);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        auto tmp5 = std::erf(tmp4);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = decltype(tmp2)(tmp2 * tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                        out_ptr0[static_cast<long>(x2 + (768L*x1) + (150528L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_native_layer_norm_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (50176L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (50176L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x2) + (50176L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 - tmp6;
                            auto tmp9 = static_cast<float>(256.0);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = static_cast<float>(1e-06);
                            auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                            auto tmp13 = 1 / std::sqrt(tmp12);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp7 * tmp14;
                            auto tmp17 = tmp15 * tmp16;
                            auto tmp19 = tmp17 + tmp18;
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1 = args
    args.clear()
    assert_size_stride(arg0_1, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg1_1, (256, ), (1, ))
    assert_size_stride(arg2_1, (256, ), (1, ))
    assert_size_stride(arg3_1, (256, ), (1, ))
    assert_size_stride(arg4_1, (1536, 256), (256, 1))
    assert_size_stride(arg5_1, (1536, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (196, 196), (196, 1))
    assert_size_stride(arg9_1, (196, ), (1, ))
    assert_size_stride(arg10_1, (256, 768), (768, 1))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (1536, 256), (256, 1))
    assert_size_stride(arg15_1, (1536, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (196, 196), (196, 1))
    assert_size_stride(arg19_1, (196, ), (1, ))
    assert_size_stride(arg20_1, (256, 768), (768, 1))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (1536, 256), (256, 1))
    assert_size_stride(arg25_1, (1536, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (196, 196), (196, 1))
    assert_size_stride(arg29_1, (196, ), (1, ))
    assert_size_stride(arg30_1, (256, 768), (768, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (1536, 256), (256, 1))
    assert_size_stride(arg35_1, (1536, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (196, 196), (196, 1))
    assert_size_stride(arg39_1, (196, ), (1, ))
    assert_size_stride(arg40_1, (256, 768), (768, 1))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (1536, 256), (256, 1))
    assert_size_stride(arg45_1, (1536, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (196, 196), (196, 1))
    assert_size_stride(arg49_1, (196, ), (1, ))
    assert_size_stride(arg50_1, (256, 768), (768, 1))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (1536, 256), (256, 1))
    assert_size_stride(arg55_1, (1536, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (196, 196), (196, 1))
    assert_size_stride(arg59_1, (196, ), (1, ))
    assert_size_stride(arg60_1, (256, 768), (768, 1))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (1536, 256), (256, 1))
    assert_size_stride(arg65_1, (1536, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (196, 196), (196, 1))
    assert_size_stride(arg69_1, (196, ), (1, ))
    assert_size_stride(arg70_1, (256, 768), (768, 1))
    assert_size_stride(arg71_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (1536, 256), (256, 1))
    assert_size_stride(arg75_1, (1536, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (196, 196), (196, 1))
    assert_size_stride(arg79_1, (196, ), (1, ))
    assert_size_stride(arg80_1, (256, 768), (768, 1))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, ), (1, ))
    assert_size_stride(arg84_1, (1536, 256), (256, 1))
    assert_size_stride(arg85_1, (1536, ), (1, ))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (196, 196), (196, 1))
    assert_size_stride(arg89_1, (196, ), (1, ))
    assert_size_stride(arg90_1, (256, 768), (768, 1))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (1536, 256), (256, 1))
    assert_size_stride(arg95_1, (1536, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (196, 196), (196, 1))
    assert_size_stride(arg99_1, (196, ), (1, ))
    assert_size_stride(arg100_1, (256, 768), (768, 1))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (1536, 256), (256, 1))
    assert_size_stride(arg105_1, (1536, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (196, 196), (196, 1))
    assert_size_stride(arg109_1, (196, ), (1, ))
    assert_size_stride(arg110_1, (256, 768), (768, 1))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (256, ), (1, ))
    assert_size_stride(arg114_1, (1536, 256), (256, 1))
    assert_size_stride(arg115_1, (1536, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (196, 196), (196, 1))
    assert_size_stride(arg119_1, (196, ), (1, ))
    assert_size_stride(arg120_1, (256, 768), (768, 1))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (1536, 256), (256, 1))
    assert_size_stride(arg125_1, (1536, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (196, 196), (196, 1))
    assert_size_stride(arg129_1, (196, ), (1, ))
    assert_size_stride(arg130_1, (256, 768), (768, 1))
    assert_size_stride(arg131_1, (256, ), (1, ))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (1536, 256), (256, 1))
    assert_size_stride(arg135_1, (1536, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (196, 196), (196, 1))
    assert_size_stride(arg139_1, (196, ), (1, ))
    assert_size_stride(arg140_1, (256, 768), (768, 1))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (1536, 256), (256, 1))
    assert_size_stride(arg145_1, (1536, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (196, 196), (196, 1))
    assert_size_stride(arg149_1, (196, ), (1, ))
    assert_size_stride(arg150_1, (256, 768), (768, 1))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (256, ), (1, ))
    assert_size_stride(arg154_1, (1536, 256), (256, 1))
    assert_size_stride(arg155_1, (1536, ), (1, ))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (196, 196), (196, 1))
    assert_size_stride(arg159_1, (196, ), (1, ))
    assert_size_stride(arg160_1, (256, 768), (768, 1))
    assert_size_stride(arg161_1, (256, ), (1, ))
    assert_size_stride(arg162_1, (256, ), (1, ))
    assert_size_stride(arg163_1, (256, ), (1, ))
    assert_size_stride(arg164_1, (1536, 256), (256, 1))
    assert_size_stride(arg165_1, (1536, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (196, 196), (196, 1))
    assert_size_stride(arg169_1, (196, ), (1, ))
    assert_size_stride(arg170_1, (256, 768), (768, 1))
    assert_size_stride(arg171_1, (256, ), (1, ))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (1536, 256), (256, 1))
    assert_size_stride(arg175_1, (1536, ), (1, ))
    assert_size_stride(arg176_1, (768, ), (1, ))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (196, 196), (196, 1))
    assert_size_stride(arg179_1, (196, ), (1, ))
    assert_size_stride(arg180_1, (256, 768), (768, 1))
    assert_size_stride(arg181_1, (256, ), (1, ))
    assert_size_stride(arg182_1, (256, ), (1, ))
    assert_size_stride(arg183_1, (256, ), (1, ))
    assert_size_stride(arg184_1, (1536, 256), (256, 1))
    assert_size_stride(arg185_1, (1536, ), (1, ))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (196, 196), (196, 1))
    assert_size_stride(arg189_1, (196, ), (1, ))
    assert_size_stride(arg190_1, (256, 768), (768, 1))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (1536, 256), (256, 1))
    assert_size_stride(arg195_1, (1536, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (196, 196), (196, 1))
    assert_size_stride(arg199_1, (196, ), (1, ))
    assert_size_stride(arg200_1, (256, 768), (768, 1))
    assert_size_stride(arg201_1, (256, ), (1, ))
    assert_size_stride(arg202_1, (256, ), (1, ))
    assert_size_stride(arg203_1, (256, ), (1, ))
    assert_size_stride(arg204_1, (1536, 256), (256, 1))
    assert_size_stride(arg205_1, (1536, ), (1, ))
    assert_size_stride(arg206_1, (768, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (196, 196), (196, 1))
    assert_size_stride(arg209_1, (196, ), (1, ))
    assert_size_stride(arg210_1, (256, 768), (768, 1))
    assert_size_stride(arg211_1, (256, ), (1, ))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (256, ), (1, ))
    assert_size_stride(arg214_1, (1536, 256), (256, 1))
    assert_size_stride(arg215_1, (1536, ), (1, ))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (196, 196), (196, 1))
    assert_size_stride(arg219_1, (196, ), (1, ))
    assert_size_stride(arg220_1, (256, 768), (768, 1))
    assert_size_stride(arg221_1, (256, ), (1, ))
    assert_size_stride(arg222_1, (256, ), (1, ))
    assert_size_stride(arg223_1, (256, ), (1, ))
    assert_size_stride(arg224_1, (1536, 256), (256, 1))
    assert_size_stride(arg225_1, (1536, ), (1, ))
    assert_size_stride(arg226_1, (768, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (196, 196), (196, 1))
    assert_size_stride(arg229_1, (196, ), (1, ))
    assert_size_stride(arg230_1, (256, 768), (768, 1))
    assert_size_stride(arg231_1, (256, ), (1, ))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (1536, 256), (256, 1))
    assert_size_stride(arg235_1, (1536, ), (1, ))
    assert_size_stride(arg236_1, (768, ), (1, ))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (196, 196), (196, 1))
    assert_size_stride(arg239_1, (196, ), (1, ))
    assert_size_stride(arg240_1, (256, 768), (768, 1))
    assert_size_stride(arg241_1, (256, ), (1, ))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (256, ), (1, ))
    assert_size_stride(arg244_1, (1536, 256), (256, 1))
    assert_size_stride(arg245_1, (1536, ), (1, ))
    assert_size_stride(arg246_1, (768, ), (1, ))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (196, 196), (196, 1))
    assert_size_stride(arg249_1, (196, ), (1, ))
    assert_size_stride(arg250_1, (256, 768), (768, 1))
    assert_size_stride(arg251_1, (256, ), (1, ))
    assert_size_stride(arg252_1, (256, ), (1, ))
    assert_size_stride(arg253_1, (256, ), (1, ))
    assert_size_stride(arg254_1, (1536, 256), (256, 1))
    assert_size_stride(arg255_1, (1536, ), (1, ))
    assert_size_stride(arg256_1, (768, ), (1, ))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (196, 196), (196, 1))
    assert_size_stride(arg259_1, (196, ), (1, ))
    assert_size_stride(arg260_1, (256, 768), (768, 1))
    assert_size_stride(arg261_1, (256, ), (1, ))
    assert_size_stride(arg262_1, (256, ), (1, ))
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (1536, 256), (256, 1))
    assert_size_stride(arg265_1, (1536, ), (1, ))
    assert_size_stride(arg266_1, (768, ), (1, ))
    assert_size_stride(arg267_1, (768, ), (1, ))
    assert_size_stride(arg268_1, (196, 196), (196, 1))
    assert_size_stride(arg269_1, (196, ), (1, ))
    assert_size_stride(arg270_1, (256, 768), (768, 1))
    assert_size_stride(arg271_1, (256, ), (1, ))
    assert_size_stride(arg272_1, (256, ), (1, ))
    assert_size_stride(arg273_1, (256, ), (1, ))
    assert_size_stride(arg274_1, (1536, 256), (256, 1))
    assert_size_stride(arg275_1, (1536, ), (1, ))
    assert_size_stride(arg276_1, (768, ), (1, ))
    assert_size_stride(arg277_1, (768, ), (1, ))
    assert_size_stride(arg278_1, (196, 196), (196, 1))
    assert_size_stride(arg279_1, (196, ), (1, ))
    assert_size_stride(arg280_1, (256, 768), (768, 1))
    assert_size_stride(arg281_1, (256, ), (1, ))
    assert_size_stride(arg282_1, (256, ), (1, ))
    assert_size_stride(arg283_1, (256, ), (1, ))
    assert_size_stride(arg284_1, (1536, 256), (256, 1))
    assert_size_stride(arg285_1, (1536, ), (1, ))
    assert_size_stride(arg286_1, (768, ), (1, ))
    assert_size_stride(arg287_1, (768, ), (1, ))
    assert_size_stride(arg288_1, (196, 196), (196, 1))
    assert_size_stride(arg289_1, (196, ), (1, ))
    assert_size_stride(arg290_1, (256, 768), (768, 1))
    assert_size_stride(arg291_1, (256, ), (1, ))
    assert_size_stride(arg292_1, (256, ), (1, ))
    assert_size_stride(arg293_1, (256, ), (1, ))
    assert_size_stride(arg294_1, (1536, 256), (256, 1))
    assert_size_stride(arg295_1, (1536, ), (1, ))
    assert_size_stride(arg296_1, (768, ), (1, ))
    assert_size_stride(arg297_1, (768, ), (1, ))
    assert_size_stride(arg298_1, (196, 196), (196, 1))
    assert_size_stride(arg299_1, (196, ), (1, ))
    assert_size_stride(arg300_1, (256, 768), (768, 1))
    assert_size_stride(arg301_1, (256, ), (1, ))
    assert_size_stride(arg302_1, (256, ), (1, ))
    assert_size_stride(arg303_1, (256, ), (1, ))
    assert_size_stride(arg304_1, (1000, 256), (256, 1))
    assert_size_stride(arg305_1, (1000, ), (1, ))
    assert_size_stride(arg306_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((256, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg306_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg306_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 256, 14, 14), (50176, 1, 3584, 256))
    del arg1_1
    del buf1
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 196, 256), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg2_1
    del arg3_1
    buf7 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg5_1, reinterpret_tensor(buf6, (1568, 256), (256, 1), 0), reinterpret_tensor(arg4_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf7)
    del arg4_1
    del arg5_1
    buf8 = buf4; del buf4  # reuse
    buf9 = buf3; del buf3  # reuse
    buf11 = reinterpret_tensor(buf0, (8, 768, 196), (150528, 196, 1), 0); del buf0  # reuse
    cpp_fused_clone_native_layer_norm_2(c_void_p(buf7.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()))
    del arg6_1
    del arg7_1
    buf12 = empty((6144, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [v_2], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (6144, 196), (196, 1), 0), reinterpret_tensor(arg8_1, (196, 196), (1, 196), 0), out=buf12)
    del arg8_1
    buf13 = reinterpret_tensor(buf11, (8, 196, 768), (150528, 768, 1), 0); del buf11  # reuse
    cpp_fused_mul_3(c_void_p(buf7.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg9_1
    buf14 = reinterpret_tensor(buf6, (1568, 256), (256, 1), 0); del buf6  # reuse
    # Source Nodes: [x_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg11_1, reinterpret_tensor(buf13, (1568, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf14)
    del arg10_1
    del arg11_1
    buf15 = buf9; del buf9  # reuse
    buf16 = buf8; del buf8  # reuse
    buf18 = empty((8, 196, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_4(c_void_p(buf2.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()))
    del arg12_1
    del arg13_1
    buf19 = buf7; del buf7  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg15_1, reinterpret_tensor(buf18, (1568, 256), (256, 1), 0), reinterpret_tensor(arg14_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf19)
    del arg14_1
    del arg15_1
    buf20 = buf16; del buf16  # reuse
    buf21 = buf15; del buf15  # reuse
    buf23 = reinterpret_tensor(buf13, (8, 768, 196), (150528, 196, 1), 0); del buf13  # reuse
    cpp_fused_clone_native_layer_norm_5(c_void_p(buf19.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg16_1
    del arg17_1
    buf24 = buf12; del buf12  # reuse
    # Source Nodes: [v_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (6144, 196), (196, 1), 0), reinterpret_tensor(arg18_1, (196, 196), (1, 196), 0), out=buf24)
    del arg18_1
    buf25 = reinterpret_tensor(buf23, (8, 196, 768), (150528, 768, 1), 0); del buf23  # reuse
    cpp_fused_mul_6(c_void_p(buf19.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf25.data_ptr()))
    del arg19_1
    buf26 = reinterpret_tensor(buf18, (1568, 256), (256, 1), 0); del buf18  # reuse
    # Source Nodes: [x_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg21_1, reinterpret_tensor(buf25, (1568, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf26)
    del arg20_1
    del arg21_1
    buf27 = buf21; del buf21  # reuse
    buf28 = buf20; del buf20  # reuse
    buf30 = empty((8, 196, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_7(c_void_p(buf2.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()))
    del arg22_1
    del arg23_1
    buf31 = buf19; del buf19  # reuse
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg25_1, reinterpret_tensor(buf30, (1568, 256), (256, 1), 0), reinterpret_tensor(arg24_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf31)
    del arg24_1
    del arg25_1
    buf32 = buf28; del buf28  # reuse
    buf33 = buf27; del buf27  # reuse
    buf35 = reinterpret_tensor(buf25, (8, 768, 196), (150528, 196, 1), 0); del buf25  # reuse
    cpp_fused_clone_native_layer_norm_8(c_void_p(buf31.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf35.data_ptr()))
    del arg26_1
    del arg27_1
    buf36 = buf24; del buf24  # reuse
    # Source Nodes: [v_8], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (6144, 196), (196, 1), 0), reinterpret_tensor(arg28_1, (196, 196), (1, 196), 0), out=buf36)
    del arg28_1
    buf37 = reinterpret_tensor(buf35, (8, 196, 768), (150528, 768, 1), 0); del buf35  # reuse
    cpp_fused_mul_9(c_void_p(buf31.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf37.data_ptr()))
    del arg29_1
    buf38 = reinterpret_tensor(buf30, (1568, 256), (256, 1), 0); del buf30  # reuse
    # Source Nodes: [x_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg31_1, reinterpret_tensor(buf37, (1568, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf38)
    del arg30_1
    del arg31_1
    buf39 = buf33; del buf33  # reuse
    buf40 = buf32; del buf32  # reuse
    buf42 = empty((8, 196, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_10(c_void_p(buf2.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()))
    del arg32_1
    del arg33_1
    buf43 = buf31; del buf31  # reuse
    # Source Nodes: [x_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg35_1, reinterpret_tensor(buf42, (1568, 256), (256, 1), 0), reinterpret_tensor(arg34_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf43)
    del arg34_1
    del arg35_1
    buf44 = buf40; del buf40  # reuse
    buf45 = buf39; del buf39  # reuse
    buf47 = reinterpret_tensor(buf37, (8, 768, 196), (150528, 196, 1), 0); del buf37  # reuse
    cpp_fused_clone_native_layer_norm_11(c_void_p(buf43.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()))
    del arg36_1
    del arg37_1
    buf48 = buf36; del buf36  # reuse
    # Source Nodes: [v_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (6144, 196), (196, 1), 0), reinterpret_tensor(arg38_1, (196, 196), (1, 196), 0), out=buf48)
    del arg38_1
    buf49 = reinterpret_tensor(buf47, (8, 196, 768), (150528, 768, 1), 0); del buf47  # reuse
    cpp_fused_mul_12(c_void_p(buf43.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf49.data_ptr()))
    del arg39_1
    buf50 = reinterpret_tensor(buf42, (1568, 256), (256, 1), 0); del buf42  # reuse
    # Source Nodes: [x_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg41_1, reinterpret_tensor(buf49, (1568, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf50)
    del arg40_1
    del arg41_1
    buf51 = reinterpret_tensor(buf50, (8, 196, 256), (50176, 256, 1), 0); del buf50  # reuse
    buf52 = buf45; del buf45  # reuse
    buf53 = buf44; del buf44  # reuse
    buf55 = empty((8, 196, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_13(c_void_p(buf51.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()))
    del arg42_1
    del arg43_1
    buf56 = buf43; del buf43  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg45_1, reinterpret_tensor(buf55, (1568, 256), (256, 1), 0), reinterpret_tensor(arg44_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf56)
    del arg44_1
    del arg45_1
    buf57 = buf53; del buf53  # reuse
    buf58 = buf52; del buf52  # reuse
    buf60 = reinterpret_tensor(buf49, (8, 768, 196), (150528, 196, 1), 0); del buf49  # reuse
    cpp_fused_clone_native_layer_norm_14(c_void_p(buf56.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()))
    del arg46_1
    del arg47_1
    buf61 = buf48; del buf48  # reuse
    # Source Nodes: [v_14], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (6144, 196), (196, 1), 0), reinterpret_tensor(arg48_1, (196, 196), (1, 196), 0), out=buf61)
    del arg48_1
    buf62 = reinterpret_tensor(buf60, (8, 196, 768), (150528, 768, 1), 0); del buf60  # reuse
    cpp_fused_mul_15(c_void_p(buf56.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(buf62.data_ptr()))
    del arg49_1
    buf63 = reinterpret_tensor(buf55, (1568, 256), (256, 1), 0); del buf55  # reuse
    # Source Nodes: [x_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg51_1, reinterpret_tensor(buf62, (1568, 768), (768, 1), 0), reinterpret_tensor(arg50_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf63)
    del arg50_1
    del arg51_1
    buf64 = buf58; del buf58  # reuse
    buf65 = buf57; del buf57  # reuse
    buf67 = reinterpret_tensor(buf38, (8, 196, 256), (50176, 256, 1), 0); del buf38  # reuse
    cpp_fused_add_native_layer_norm_16(c_void_p(buf51.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()))
    del arg52_1
    del arg53_1
    buf68 = buf56; del buf56  # reuse
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf67, (1568, 256), (256, 1), 0), reinterpret_tensor(arg54_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf68)
    del arg54_1
    del arg55_1
    buf69 = buf65; del buf65  # reuse
    buf70 = buf64; del buf64  # reuse
    buf72 = reinterpret_tensor(buf62, (8, 768, 196), (150528, 196, 1), 0); del buf62  # reuse
    cpp_fused_clone_native_layer_norm_17(c_void_p(buf68.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()))
    del arg56_1
    del arg57_1
    buf73 = buf61; del buf61  # reuse
    # Source Nodes: [v_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (6144, 196), (196, 1), 0), reinterpret_tensor(arg58_1, (196, 196), (1, 196), 0), out=buf73)
    del arg58_1
    buf74 = reinterpret_tensor(buf72, (8, 196, 768), (150528, 768, 1), 0); del buf72  # reuse
    cpp_fused_mul_18(c_void_p(buf68.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf74.data_ptr()))
    del arg59_1
    buf75 = reinterpret_tensor(buf67, (1568, 256), (256, 1), 0); del buf67  # reuse
    # Source Nodes: [x_49], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg61_1, reinterpret_tensor(buf74, (1568, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf75)
    del arg60_1
    del arg61_1
    buf76 = buf70; del buf70  # reuse
    buf77 = buf69; del buf69  # reuse
    buf79 = reinterpret_tensor(buf26, (8, 196, 256), (50176, 256, 1), 0); del buf26  # reuse
    cpp_fused_add_native_layer_norm_19(c_void_p(buf51.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()))
    del arg62_1
    del arg63_1
    buf80 = buf68; del buf68  # reuse
    # Source Nodes: [x_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg65_1, reinterpret_tensor(buf79, (1568, 256), (256, 1), 0), reinterpret_tensor(arg64_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf80)
    del arg64_1
    del arg65_1
    buf81 = buf77; del buf77  # reuse
    buf82 = buf76; del buf76  # reuse
    buf84 = reinterpret_tensor(buf74, (8, 768, 196), (150528, 196, 1), 0); del buf74  # reuse
    cpp_fused_clone_native_layer_norm_20(c_void_p(buf80.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()))
    del arg66_1
    del arg67_1
    buf85 = buf73; del buf73  # reuse
    # Source Nodes: [v_20], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf84, (6144, 196), (196, 1), 0), reinterpret_tensor(arg68_1, (196, 196), (1, 196), 0), out=buf85)
    del arg68_1
    buf86 = reinterpret_tensor(buf84, (8, 196, 768), (150528, 768, 1), 0); del buf84  # reuse
    cpp_fused_mul_21(c_void_p(buf80.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf86.data_ptr()))
    del arg69_1
    buf87 = reinterpret_tensor(buf79, (1568, 256), (256, 1), 0); del buf79  # reuse
    # Source Nodes: [x_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf86, (1568, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf87)
    del arg70_1
    del arg71_1
    buf88 = buf82; del buf82  # reuse
    buf89 = buf81; del buf81  # reuse
    buf91 = reinterpret_tensor(buf2, (8, 196, 256), (50176, 256, 1), 0); del buf2  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf51.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf91.data_ptr()))
    del arg72_1
    del arg73_1
    buf92 = buf80; del buf80  # reuse
    # Source Nodes: [x_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg75_1, reinterpret_tensor(buf91, (1568, 256), (256, 1), 0), reinterpret_tensor(arg74_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf92)
    del arg74_1
    del arg75_1
    buf93 = buf89; del buf89  # reuse
    buf94 = buf88; del buf88  # reuse
    buf96 = reinterpret_tensor(buf86, (8, 768, 196), (150528, 196, 1), 0); del buf86  # reuse
    cpp_fused_clone_native_layer_norm_23(c_void_p(buf92.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()))
    del arg76_1
    del arg77_1
    buf97 = buf85; del buf85  # reuse
    # Source Nodes: [v_23], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (6144, 196), (196, 1), 0), reinterpret_tensor(arg78_1, (196, 196), (1, 196), 0), out=buf97)
    del arg78_1
    buf98 = reinterpret_tensor(buf96, (8, 196, 768), (150528, 768, 1), 0); del buf96  # reuse
    cpp_fused_mul_24(c_void_p(buf92.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(buf98.data_ptr()))
    del arg79_1
    buf99 = reinterpret_tensor(buf91, (1568, 256), (256, 1), 0); del buf91  # reuse
    # Source Nodes: [x_65], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg81_1, reinterpret_tensor(buf98, (1568, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf99)
    del arg80_1
    del arg81_1
    buf100 = reinterpret_tensor(buf99, (8, 196, 256), (50176, 256, 1), 0); del buf99  # reuse
    buf101 = buf94; del buf94  # reuse
    buf102 = buf93; del buf93  # reuse
    buf104 = reinterpret_tensor(buf14, (8, 196, 256), (50176, 256, 1), 0); del buf14  # reuse
    cpp_fused_add_native_layer_norm_25(c_void_p(buf100.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()))
    del arg82_1
    del arg83_1
    buf105 = buf92; del buf92  # reuse
    # Source Nodes: [x_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf104, (1568, 256), (256, 1), 0), reinterpret_tensor(arg84_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf105)
    del arg84_1
    del arg85_1
    buf106 = buf102; del buf102  # reuse
    buf107 = buf101; del buf101  # reuse
    buf109 = reinterpret_tensor(buf98, (8, 768, 196), (150528, 196, 1), 0); del buf98  # reuse
    cpp_fused_clone_native_layer_norm_26(c_void_p(buf105.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg86_1
    del arg87_1
    buf110 = buf97; del buf97  # reuse
    # Source Nodes: [v_26], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf109, (6144, 196), (196, 1), 0), reinterpret_tensor(arg88_1, (196, 196), (1, 196), 0), out=buf110)
    del arg88_1
    buf111 = reinterpret_tensor(buf109, (8, 196, 768), (150528, 768, 1), 0); del buf109  # reuse
    cpp_fused_mul_27(c_void_p(buf105.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(buf111.data_ptr()))
    del arg89_1
    buf112 = reinterpret_tensor(buf104, (1568, 256), (256, 1), 0); del buf104  # reuse
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf111, (1568, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf112)
    del arg90_1
    del arg91_1
    buf113 = buf107; del buf107  # reuse
    buf114 = buf106; del buf106  # reuse
    buf116 = reinterpret_tensor(buf87, (8, 196, 256), (50176, 256, 1), 0); del buf87  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf100.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    del arg92_1
    del arg93_1
    buf117 = buf105; del buf105  # reuse
    # Source Nodes: [x_76], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, reinterpret_tensor(buf116, (1568, 256), (256, 1), 0), reinterpret_tensor(arg94_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf117)
    del arg94_1
    del arg95_1
    buf118 = buf114; del buf114  # reuse
    buf119 = buf113; del buf113  # reuse
    buf121 = reinterpret_tensor(buf111, (8, 768, 196), (150528, 196, 1), 0); del buf111  # reuse
    cpp_fused_clone_native_layer_norm_29(c_void_p(buf117.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()))
    del arg96_1
    del arg97_1
    buf122 = buf110; del buf110  # reuse
    # Source Nodes: [v_29], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf121, (6144, 196), (196, 1), 0), reinterpret_tensor(arg98_1, (196, 196), (1, 196), 0), out=buf122)
    del arg98_1
    buf123 = reinterpret_tensor(buf121, (8, 196, 768), (150528, 768, 1), 0); del buf121  # reuse
    cpp_fused_mul_30(c_void_p(buf117.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf123.data_ptr()))
    del arg99_1
    buf124 = reinterpret_tensor(buf116, (1568, 256), (256, 1), 0); del buf116  # reuse
    # Source Nodes: [x_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf123, (1568, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf124)
    del arg100_1
    del arg101_1
    buf125 = buf119; del buf119  # reuse
    buf126 = buf118; del buf118  # reuse
    buf128 = reinterpret_tensor(buf75, (8, 196, 256), (50176, 256, 1), 0); del buf75  # reuse
    cpp_fused_add_native_layer_norm_31(c_void_p(buf100.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()))
    del arg102_1
    del arg103_1
    buf129 = buf117; del buf117  # reuse
    # Source Nodes: [x_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf128, (1568, 256), (256, 1), 0), reinterpret_tensor(arg104_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf129)
    del arg104_1
    del arg105_1
    buf130 = buf126; del buf126  # reuse
    buf131 = buf125; del buf125  # reuse
    buf133 = reinterpret_tensor(buf123, (8, 768, 196), (150528, 196, 1), 0); del buf123  # reuse
    cpp_fused_clone_native_layer_norm_32(c_void_p(buf129.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf133.data_ptr()))
    del arg106_1
    del arg107_1
    buf134 = buf122; del buf122  # reuse
    # Source Nodes: [v_32], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf133, (6144, 196), (196, 1), 0), reinterpret_tensor(arg108_1, (196, 196), (1, 196), 0), out=buf134)
    del arg108_1
    buf135 = reinterpret_tensor(buf133, (8, 196, 768), (150528, 768, 1), 0); del buf133  # reuse
    cpp_fused_mul_33(c_void_p(buf129.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg109_1
    buf136 = reinterpret_tensor(buf128, (1568, 256), (256, 1), 0); del buf128  # reuse
    # Source Nodes: [x_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg111_1, reinterpret_tensor(buf135, (1568, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf136)
    del arg110_1
    del arg111_1
    buf137 = buf131; del buf131  # reuse
    buf138 = buf130; del buf130  # reuse
    buf140 = reinterpret_tensor(buf63, (8, 196, 256), (50176, 256, 1), 0); del buf63  # reuse
    cpp_fused_add_native_layer_norm_34(c_void_p(buf100.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()))
    del arg112_1
    del arg113_1
    buf141 = buf129; del buf129  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf140, (1568, 256), (256, 1), 0), reinterpret_tensor(arg114_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf141)
    del arg114_1
    del arg115_1
    buf142 = buf138; del buf138  # reuse
    buf143 = buf137; del buf137  # reuse
    buf145 = reinterpret_tensor(buf135, (8, 768, 196), (150528, 196, 1), 0); del buf135  # reuse
    cpp_fused_clone_native_layer_norm_35(c_void_p(buf141.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf145.data_ptr()))
    del arg116_1
    del arg117_1
    buf146 = buf134; del buf134  # reuse
    # Source Nodes: [v_35], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (6144, 196), (196, 1), 0), reinterpret_tensor(arg118_1, (196, 196), (1, 196), 0), out=buf146)
    del arg118_1
    buf147 = reinterpret_tensor(buf145, (8, 196, 768), (150528, 768, 1), 0); del buf145  # reuse
    cpp_fused_mul_36(c_void_p(buf141.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf147.data_ptr()))
    del arg119_1
    buf148 = reinterpret_tensor(buf140, (1568, 256), (256, 1), 0); del buf140  # reuse
    # Source Nodes: [x_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf147, (1568, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf148)
    del arg120_1
    del arg121_1
    buf149 = reinterpret_tensor(buf148, (8, 196, 256), (50176, 256, 1), 0); del buf148  # reuse
    buf150 = buf143; del buf143  # reuse
    buf151 = buf142; del buf142  # reuse
    buf153 = buf51; del buf51  # reuse
    cpp_fused_add_native_layer_norm_37(c_void_p(buf149.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg122_1
    del arg123_1
    buf154 = buf141; del buf141  # reuse
    # Source Nodes: [x_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg125_1, reinterpret_tensor(buf153, (1568, 256), (256, 1), 0), reinterpret_tensor(arg124_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf154)
    del arg124_1
    del arg125_1
    buf155 = buf151; del buf151  # reuse
    buf156 = buf150; del buf150  # reuse
    buf158 = reinterpret_tensor(buf147, (8, 768, 196), (150528, 196, 1), 0); del buf147  # reuse
    cpp_fused_clone_native_layer_norm_38(c_void_p(buf154.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()))
    del arg126_1
    del arg127_1
    buf159 = buf146; del buf146  # reuse
    # Source Nodes: [v_38], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (6144, 196), (196, 1), 0), reinterpret_tensor(arg128_1, (196, 196), (1, 196), 0), out=buf159)
    del arg128_1
    buf160 = reinterpret_tensor(buf158, (8, 196, 768), (150528, 768, 1), 0); del buf158  # reuse
    cpp_fused_mul_39(c_void_p(buf154.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(buf160.data_ptr()))
    del arg129_1
    buf161 = reinterpret_tensor(buf153, (1568, 256), (256, 1), 0); del buf153  # reuse
    # Source Nodes: [x_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf160, (1568, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf161)
    del arg130_1
    del arg131_1
    buf162 = buf156; del buf156  # reuse
    buf163 = buf155; del buf155  # reuse
    buf165 = reinterpret_tensor(buf136, (8, 196, 256), (50176, 256, 1), 0); del buf136  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf149.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()))
    del arg132_1
    del arg133_1
    buf166 = buf154; del buf154  # reuse
    # Source Nodes: [x_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg135_1, reinterpret_tensor(buf165, (1568, 256), (256, 1), 0), reinterpret_tensor(arg134_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf166)
    del arg134_1
    del arg135_1
    buf167 = buf163; del buf163  # reuse
    buf168 = buf162; del buf162  # reuse
    buf170 = reinterpret_tensor(buf160, (8, 768, 196), (150528, 196, 1), 0); del buf160  # reuse
    cpp_fused_clone_native_layer_norm_41(c_void_p(buf166.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()))
    del arg136_1
    del arg137_1
    buf171 = buf159; del buf159  # reuse
    # Source Nodes: [v_41], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf170, (6144, 196), (196, 1), 0), reinterpret_tensor(arg138_1, (196, 196), (1, 196), 0), out=buf171)
    del arg138_1
    buf172 = reinterpret_tensor(buf170, (8, 196, 768), (150528, 768, 1), 0); del buf170  # reuse
    cpp_fused_mul_42(c_void_p(buf166.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(buf172.data_ptr()))
    del arg139_1
    buf173 = reinterpret_tensor(buf165, (1568, 256), (256, 1), 0); del buf165  # reuse
    # Source Nodes: [x_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg141_1, reinterpret_tensor(buf172, (1568, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf173)
    del arg140_1
    del arg141_1
    buf174 = buf168; del buf168  # reuse
    buf175 = buf167; del buf167  # reuse
    buf177 = reinterpret_tensor(buf124, (8, 196, 256), (50176, 256, 1), 0); del buf124  # reuse
    cpp_fused_add_native_layer_norm_43(c_void_p(buf149.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf177.data_ptr()))
    del arg142_1
    del arg143_1
    buf178 = buf166; del buf166  # reuse
    # Source Nodes: [x_116], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf177, (1568, 256), (256, 1), 0), reinterpret_tensor(arg144_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf178)
    del arg144_1
    del arg145_1
    buf179 = buf175; del buf175  # reuse
    buf180 = buf174; del buf174  # reuse
    buf182 = reinterpret_tensor(buf172, (8, 768, 196), (150528, 196, 1), 0); del buf172  # reuse
    cpp_fused_clone_native_layer_norm_44(c_void_p(buf178.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()))
    del arg146_1
    del arg147_1
    buf183 = buf171; del buf171  # reuse
    # Source Nodes: [v_44], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (6144, 196), (196, 1), 0), reinterpret_tensor(arg148_1, (196, 196), (1, 196), 0), out=buf183)
    del arg148_1
    buf184 = reinterpret_tensor(buf182, (8, 196, 768), (150528, 768, 1), 0); del buf182  # reuse
    cpp_fused_mul_45(c_void_p(buf178.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(buf184.data_ptr()))
    del arg149_1
    buf185 = reinterpret_tensor(buf177, (1568, 256), (256, 1), 0); del buf177  # reuse
    # Source Nodes: [x_121], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg151_1, reinterpret_tensor(buf184, (1568, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf185)
    del arg150_1
    del arg151_1
    buf186 = buf180; del buf180  # reuse
    buf187 = buf179; del buf179  # reuse
    buf189 = reinterpret_tensor(buf112, (8, 196, 256), (50176, 256, 1), 0); del buf112  # reuse
    cpp_fused_add_native_layer_norm_46(c_void_p(buf149.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()))
    del arg152_1
    del arg153_1
    buf190 = buf178; del buf178  # reuse
    # Source Nodes: [x_124], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg155_1, reinterpret_tensor(buf189, (1568, 256), (256, 1), 0), reinterpret_tensor(arg154_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf190)
    del arg154_1
    del arg155_1
    buf191 = buf187; del buf187  # reuse
    buf192 = buf186; del buf186  # reuse
    buf194 = reinterpret_tensor(buf184, (8, 768, 196), (150528, 196, 1), 0); del buf184  # reuse
    cpp_fused_clone_native_layer_norm_47(c_void_p(buf190.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf194.data_ptr()))
    del arg156_1
    del arg157_1
    buf195 = buf183; del buf183  # reuse
    # Source Nodes: [v_47], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (6144, 196), (196, 1), 0), reinterpret_tensor(arg158_1, (196, 196), (1, 196), 0), out=buf195)
    del arg158_1
    buf196 = reinterpret_tensor(buf194, (8, 196, 768), (150528, 768, 1), 0); del buf194  # reuse
    cpp_fused_mul_48(c_void_p(buf190.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(buf196.data_ptr()))
    del arg159_1
    buf197 = reinterpret_tensor(buf189, (1568, 256), (256, 1), 0); del buf189  # reuse
    # Source Nodes: [x_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg161_1, reinterpret_tensor(buf196, (1568, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf197)
    del arg160_1
    del arg161_1
    buf198 = reinterpret_tensor(buf197, (8, 196, 256), (50176, 256, 1), 0); del buf197  # reuse
    buf199 = buf192; del buf192  # reuse
    buf200 = buf191; del buf191  # reuse
    buf202 = buf100; del buf100  # reuse
    cpp_fused_add_native_layer_norm_49(c_void_p(buf198.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf202.data_ptr()))
    del arg162_1
    del arg163_1
    buf203 = buf190; del buf190  # reuse
    # Source Nodes: [x_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg165_1, reinterpret_tensor(buf202, (1568, 256), (256, 1), 0), reinterpret_tensor(arg164_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf203)
    del arg164_1
    del arg165_1
    buf204 = buf200; del buf200  # reuse
    buf205 = buf199; del buf199  # reuse
    buf207 = reinterpret_tensor(buf196, (8, 768, 196), (150528, 196, 1), 0); del buf196  # reuse
    cpp_fused_clone_native_layer_norm_50(c_void_p(buf203.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()))
    del arg166_1
    del arg167_1
    buf208 = buf195; del buf195  # reuse
    # Source Nodes: [v_50], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (6144, 196), (196, 1), 0), reinterpret_tensor(arg168_1, (196, 196), (1, 196), 0), out=buf208)
    del arg168_1
    buf209 = reinterpret_tensor(buf207, (8, 196, 768), (150528, 768, 1), 0); del buf207  # reuse
    cpp_fused_mul_51(c_void_p(buf203.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(buf209.data_ptr()))
    del arg169_1
    buf210 = reinterpret_tensor(buf202, (1568, 256), (256, 1), 0); del buf202  # reuse
    # Source Nodes: [x_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg171_1, reinterpret_tensor(buf209, (1568, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf210)
    del arg170_1
    del arg171_1
    buf211 = buf205; del buf205  # reuse
    buf212 = buf204; del buf204  # reuse
    buf214 = reinterpret_tensor(buf185, (8, 196, 256), (50176, 256, 1), 0); del buf185  # reuse
    cpp_fused_add_native_layer_norm_52(c_void_p(buf198.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf214.data_ptr()))
    del arg172_1
    del arg173_1
    buf215 = buf203; del buf203  # reuse
    # Source Nodes: [x_140], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg175_1, reinterpret_tensor(buf214, (1568, 256), (256, 1), 0), reinterpret_tensor(arg174_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf215)
    del arg174_1
    del arg175_1
    buf216 = buf212; del buf212  # reuse
    buf217 = buf211; del buf211  # reuse
    buf219 = reinterpret_tensor(buf209, (8, 768, 196), (150528, 196, 1), 0); del buf209  # reuse
    cpp_fused_clone_native_layer_norm_53(c_void_p(buf215.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()))
    del arg176_1
    del arg177_1
    buf220 = buf208; del buf208  # reuse
    # Source Nodes: [v_53], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf219, (6144, 196), (196, 1), 0), reinterpret_tensor(arg178_1, (196, 196), (1, 196), 0), out=buf220)
    del arg178_1
    buf221 = reinterpret_tensor(buf219, (8, 196, 768), (150528, 768, 1), 0); del buf219  # reuse
    cpp_fused_mul_54(c_void_p(buf215.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(buf221.data_ptr()))
    del arg179_1
    buf222 = reinterpret_tensor(buf214, (1568, 256), (256, 1), 0); del buf214  # reuse
    # Source Nodes: [x_145], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg181_1, reinterpret_tensor(buf221, (1568, 768), (768, 1), 0), reinterpret_tensor(arg180_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf222)
    del arg180_1
    del arg181_1
    buf223 = buf217; del buf217  # reuse
    buf224 = buf216; del buf216  # reuse
    buf226 = reinterpret_tensor(buf173, (8, 196, 256), (50176, 256, 1), 0); del buf173  # reuse
    cpp_fused_add_native_layer_norm_55(c_void_p(buf198.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()))
    del arg182_1
    del arg183_1
    buf227 = buf215; del buf215  # reuse
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg185_1, reinterpret_tensor(buf226, (1568, 256), (256, 1), 0), reinterpret_tensor(arg184_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf227)
    del arg184_1
    del arg185_1
    buf228 = buf224; del buf224  # reuse
    buf229 = buf223; del buf223  # reuse
    buf231 = reinterpret_tensor(buf221, (8, 768, 196), (150528, 196, 1), 0); del buf221  # reuse
    cpp_fused_clone_native_layer_norm_56(c_void_p(buf227.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()))
    del arg186_1
    del arg187_1
    buf232 = buf220; del buf220  # reuse
    # Source Nodes: [v_56], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (6144, 196), (196, 1), 0), reinterpret_tensor(arg188_1, (196, 196), (1, 196), 0), out=buf232)
    del arg188_1
    buf233 = reinterpret_tensor(buf231, (8, 196, 768), (150528, 768, 1), 0); del buf231  # reuse
    cpp_fused_mul_57(c_void_p(buf227.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(buf233.data_ptr()))
    del arg189_1
    buf234 = reinterpret_tensor(buf226, (1568, 256), (256, 1), 0); del buf226  # reuse
    # Source Nodes: [x_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg191_1, reinterpret_tensor(buf233, (1568, 768), (768, 1), 0), reinterpret_tensor(arg190_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf234)
    del arg190_1
    del arg191_1
    buf235 = buf229; del buf229  # reuse
    buf236 = buf228; del buf228  # reuse
    buf238 = reinterpret_tensor(buf161, (8, 196, 256), (50176, 256, 1), 0); del buf161  # reuse
    cpp_fused_add_native_layer_norm_58(c_void_p(buf198.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()))
    del arg192_1
    del arg193_1
    buf239 = buf227; del buf227  # reuse
    # Source Nodes: [x_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg195_1, reinterpret_tensor(buf238, (1568, 256), (256, 1), 0), reinterpret_tensor(arg194_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf239)
    del arg194_1
    del arg195_1
    buf240 = buf236; del buf236  # reuse
    buf241 = buf235; del buf235  # reuse
    buf243 = reinterpret_tensor(buf233, (8, 768, 196), (150528, 196, 1), 0); del buf233  # reuse
    cpp_fused_clone_native_layer_norm_59(c_void_p(buf239.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()))
    del arg196_1
    del arg197_1
    buf244 = buf232; del buf232  # reuse
    # Source Nodes: [v_59], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (6144, 196), (196, 1), 0), reinterpret_tensor(arg198_1, (196, 196), (1, 196), 0), out=buf244)
    del arg198_1
    buf245 = reinterpret_tensor(buf243, (8, 196, 768), (150528, 768, 1), 0); del buf243  # reuse
    cpp_fused_mul_60(c_void_p(buf239.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(buf245.data_ptr()))
    del arg199_1
    buf246 = reinterpret_tensor(buf238, (1568, 256), (256, 1), 0); del buf238  # reuse
    # Source Nodes: [x_161], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg201_1, reinterpret_tensor(buf245, (1568, 768), (768, 1), 0), reinterpret_tensor(arg200_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf246)
    del arg200_1
    del arg201_1
    buf247 = reinterpret_tensor(buf246, (8, 196, 256), (50176, 256, 1), 0); del buf246  # reuse
    buf248 = buf241; del buf241  # reuse
    buf249 = buf240; del buf240  # reuse
    buf251 = buf149; del buf149  # reuse
    cpp_fused_add_native_layer_norm_61(c_void_p(buf247.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()))
    del arg202_1
    del arg203_1
    buf252 = buf239; del buf239  # reuse
    # Source Nodes: [x_164], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg205_1, reinterpret_tensor(buf251, (1568, 256), (256, 1), 0), reinterpret_tensor(arg204_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf252)
    del arg204_1
    del arg205_1
    buf253 = buf249; del buf249  # reuse
    buf254 = buf248; del buf248  # reuse
    buf256 = reinterpret_tensor(buf245, (8, 768, 196), (150528, 196, 1), 0); del buf245  # reuse
    cpp_fused_clone_native_layer_norm_62(c_void_p(buf252.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf256.data_ptr()))
    del arg206_1
    del arg207_1
    buf257 = buf244; del buf244  # reuse
    # Source Nodes: [v_62], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf256, (6144, 196), (196, 1), 0), reinterpret_tensor(arg208_1, (196, 196), (1, 196), 0), out=buf257)
    del arg208_1
    buf258 = reinterpret_tensor(buf256, (8, 196, 768), (150528, 768, 1), 0); del buf256  # reuse
    cpp_fused_mul_63(c_void_p(buf252.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(buf258.data_ptr()))
    del arg209_1
    buf259 = reinterpret_tensor(buf251, (1568, 256), (256, 1), 0); del buf251  # reuse
    # Source Nodes: [x_169], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg211_1, reinterpret_tensor(buf258, (1568, 768), (768, 1), 0), reinterpret_tensor(arg210_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf259)
    del arg210_1
    del arg211_1
    buf260 = buf254; del buf254  # reuse
    buf261 = buf253; del buf253  # reuse
    buf263 = reinterpret_tensor(buf234, (8, 196, 256), (50176, 256, 1), 0); del buf234  # reuse
    cpp_fused_add_native_layer_norm_64(c_void_p(buf247.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf263.data_ptr()))
    del arg212_1
    del arg213_1
    buf264 = buf252; del buf252  # reuse
    # Source Nodes: [x_172], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg215_1, reinterpret_tensor(buf263, (1568, 256), (256, 1), 0), reinterpret_tensor(arg214_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf264)
    del arg214_1
    del arg215_1
    buf265 = buf261; del buf261  # reuse
    buf266 = buf260; del buf260  # reuse
    buf268 = reinterpret_tensor(buf258, (8, 768, 196), (150528, 196, 1), 0); del buf258  # reuse
    cpp_fused_clone_native_layer_norm_65(c_void_p(buf264.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()))
    del arg216_1
    del arg217_1
    buf269 = buf257; del buf257  # reuse
    # Source Nodes: [v_65], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf268, (6144, 196), (196, 1), 0), reinterpret_tensor(arg218_1, (196, 196), (1, 196), 0), out=buf269)
    del arg218_1
    buf270 = reinterpret_tensor(buf268, (8, 196, 768), (150528, 768, 1), 0); del buf268  # reuse
    cpp_fused_mul_66(c_void_p(buf264.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(buf270.data_ptr()))
    del arg219_1
    buf271 = reinterpret_tensor(buf263, (1568, 256), (256, 1), 0); del buf263  # reuse
    # Source Nodes: [x_177], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg221_1, reinterpret_tensor(buf270, (1568, 768), (768, 1), 0), reinterpret_tensor(arg220_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf271)
    del arg220_1
    del arg221_1
    buf272 = buf266; del buf266  # reuse
    buf273 = buf265; del buf265  # reuse
    buf275 = reinterpret_tensor(buf222, (8, 196, 256), (50176, 256, 1), 0); del buf222  # reuse
    cpp_fused_add_native_layer_norm_67(c_void_p(buf247.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf275.data_ptr()))
    del arg222_1
    del arg223_1
    buf276 = buf264; del buf264  # reuse
    # Source Nodes: [x_180], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg225_1, reinterpret_tensor(buf275, (1568, 256), (256, 1), 0), reinterpret_tensor(arg224_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf276)
    del arg224_1
    del arg225_1
    buf277 = buf273; del buf273  # reuse
    buf278 = buf272; del buf272  # reuse
    buf280 = reinterpret_tensor(buf270, (8, 768, 196), (150528, 196, 1), 0); del buf270  # reuse
    cpp_fused_clone_native_layer_norm_68(c_void_p(buf276.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()))
    del arg226_1
    del arg227_1
    buf281 = buf269; del buf269  # reuse
    # Source Nodes: [v_68], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf280, (6144, 196), (196, 1), 0), reinterpret_tensor(arg228_1, (196, 196), (1, 196), 0), out=buf281)
    del arg228_1
    buf282 = reinterpret_tensor(buf280, (8, 196, 768), (150528, 768, 1), 0); del buf280  # reuse
    cpp_fused_mul_69(c_void_p(buf276.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(buf282.data_ptr()))
    del arg229_1
    buf283 = reinterpret_tensor(buf275, (1568, 256), (256, 1), 0); del buf275  # reuse
    # Source Nodes: [x_185], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg231_1, reinterpret_tensor(buf282, (1568, 768), (768, 1), 0), reinterpret_tensor(arg230_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf283)
    del arg230_1
    del arg231_1
    buf284 = buf278; del buf278  # reuse
    buf285 = buf277; del buf277  # reuse
    buf287 = reinterpret_tensor(buf210, (8, 196, 256), (50176, 256, 1), 0); del buf210  # reuse
    cpp_fused_add_native_layer_norm_70(c_void_p(buf247.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf287.data_ptr()))
    del arg232_1
    del arg233_1
    buf288 = buf276; del buf276  # reuse
    # Source Nodes: [x_188], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg235_1, reinterpret_tensor(buf287, (1568, 256), (256, 1), 0), reinterpret_tensor(arg234_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf288)
    del arg234_1
    del arg235_1
    buf289 = buf285; del buf285  # reuse
    buf290 = buf284; del buf284  # reuse
    buf292 = reinterpret_tensor(buf282, (8, 768, 196), (150528, 196, 1), 0); del buf282  # reuse
    cpp_fused_clone_native_layer_norm_71(c_void_p(buf288.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf292.data_ptr()))
    del arg236_1
    del arg237_1
    buf293 = buf281; del buf281  # reuse
    # Source Nodes: [v_71], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (6144, 196), (196, 1), 0), reinterpret_tensor(arg238_1, (196, 196), (1, 196), 0), out=buf293)
    del arg238_1
    buf294 = reinterpret_tensor(buf292, (8, 196, 768), (150528, 768, 1), 0); del buf292  # reuse
    cpp_fused_mul_72(c_void_p(buf288.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(buf294.data_ptr()))
    del arg239_1
    buf295 = reinterpret_tensor(buf287, (1568, 256), (256, 1), 0); del buf287  # reuse
    # Source Nodes: [x_193], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg241_1, reinterpret_tensor(buf294, (1568, 768), (768, 1), 0), reinterpret_tensor(arg240_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf295)
    del arg240_1
    del arg241_1
    buf296 = reinterpret_tensor(buf295, (8, 196, 256), (50176, 256, 1), 0); del buf295  # reuse
    buf297 = buf290; del buf290  # reuse
    buf298 = buf289; del buf289  # reuse
    buf300 = buf198; del buf198  # reuse
    cpp_fused_add_native_layer_norm_73(c_void_p(buf296.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()))
    del arg242_1
    del arg243_1
    buf301 = buf288; del buf288  # reuse
    # Source Nodes: [x_196], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg245_1, reinterpret_tensor(buf300, (1568, 256), (256, 1), 0), reinterpret_tensor(arg244_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf301)
    del arg244_1
    del arg245_1
    buf302 = buf298; del buf298  # reuse
    buf303 = buf297; del buf297  # reuse
    buf305 = reinterpret_tensor(buf294, (8, 768, 196), (150528, 196, 1), 0); del buf294  # reuse
    cpp_fused_clone_native_layer_norm_74(c_void_p(buf301.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf305.data_ptr()))
    del arg246_1
    del arg247_1
    buf306 = buf293; del buf293  # reuse
    # Source Nodes: [v_74], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf305, (6144, 196), (196, 1), 0), reinterpret_tensor(arg248_1, (196, 196), (1, 196), 0), out=buf306)
    del arg248_1
    buf307 = reinterpret_tensor(buf305, (8, 196, 768), (150528, 768, 1), 0); del buf305  # reuse
    cpp_fused_mul_75(c_void_p(buf301.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(buf307.data_ptr()))
    del arg249_1
    buf308 = reinterpret_tensor(buf300, (1568, 256), (256, 1), 0); del buf300  # reuse
    # Source Nodes: [x_201], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg251_1, reinterpret_tensor(buf307, (1568, 768), (768, 1), 0), reinterpret_tensor(arg250_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf308)
    del arg250_1
    del arg251_1
    buf309 = buf303; del buf303  # reuse
    buf310 = buf302; del buf302  # reuse
    buf312 = reinterpret_tensor(buf283, (8, 196, 256), (50176, 256, 1), 0); del buf283  # reuse
    cpp_fused_add_native_layer_norm_76(c_void_p(buf296.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()))
    del arg252_1
    del arg253_1
    buf313 = buf301; del buf301  # reuse
    # Source Nodes: [x_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg255_1, reinterpret_tensor(buf312, (1568, 256), (256, 1), 0), reinterpret_tensor(arg254_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf313)
    del arg254_1
    del arg255_1
    buf314 = buf310; del buf310  # reuse
    buf315 = buf309; del buf309  # reuse
    buf317 = reinterpret_tensor(buf307, (8, 768, 196), (150528, 196, 1), 0); del buf307  # reuse
    cpp_fused_clone_native_layer_norm_77(c_void_p(buf313.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf317.data_ptr()))
    del arg256_1
    del arg257_1
    buf318 = buf306; del buf306  # reuse
    # Source Nodes: [v_77], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (6144, 196), (196, 1), 0), reinterpret_tensor(arg258_1, (196, 196), (1, 196), 0), out=buf318)
    del arg258_1
    buf319 = reinterpret_tensor(buf317, (8, 196, 768), (150528, 768, 1), 0); del buf317  # reuse
    cpp_fused_mul_78(c_void_p(buf313.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(buf319.data_ptr()))
    del arg259_1
    buf320 = reinterpret_tensor(buf312, (1568, 256), (256, 1), 0); del buf312  # reuse
    # Source Nodes: [x_209], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg261_1, reinterpret_tensor(buf319, (1568, 768), (768, 1), 0), reinterpret_tensor(arg260_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf320)
    del arg260_1
    del arg261_1
    buf321 = buf315; del buf315  # reuse
    buf322 = buf314; del buf314  # reuse
    buf324 = reinterpret_tensor(buf271, (8, 196, 256), (50176, 256, 1), 0); del buf271  # reuse
    cpp_fused_add_native_layer_norm_79(c_void_p(buf296.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()))
    del arg262_1
    del arg263_1
    buf325 = buf313; del buf313  # reuse
    # Source Nodes: [x_212], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg265_1, reinterpret_tensor(buf324, (1568, 256), (256, 1), 0), reinterpret_tensor(arg264_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf325)
    del arg264_1
    del arg265_1
    buf326 = buf322; del buf322  # reuse
    buf327 = buf321; del buf321  # reuse
    buf329 = reinterpret_tensor(buf319, (8, 768, 196), (150528, 196, 1), 0); del buf319  # reuse
    cpp_fused_clone_native_layer_norm_80(c_void_p(buf325.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf329.data_ptr()))
    del arg266_1
    del arg267_1
    buf330 = buf318; del buf318  # reuse
    # Source Nodes: [v_80], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf329, (6144, 196), (196, 1), 0), reinterpret_tensor(arg268_1, (196, 196), (1, 196), 0), out=buf330)
    del arg268_1
    buf331 = reinterpret_tensor(buf329, (8, 196, 768), (150528, 768, 1), 0); del buf329  # reuse
    cpp_fused_mul_81(c_void_p(buf325.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(buf331.data_ptr()))
    del arg269_1
    buf332 = reinterpret_tensor(buf324, (1568, 256), (256, 1), 0); del buf324  # reuse
    # Source Nodes: [x_217], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg271_1, reinterpret_tensor(buf331, (1568, 768), (768, 1), 0), reinterpret_tensor(arg270_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf332)
    del arg270_1
    del arg271_1
    buf333 = buf327; del buf327  # reuse
    buf334 = buf326; del buf326  # reuse
    buf336 = reinterpret_tensor(buf259, (8, 196, 256), (50176, 256, 1), 0); del buf259  # reuse
    cpp_fused_add_native_layer_norm_82(c_void_p(buf296.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()))
    del arg272_1
    del arg273_1
    buf337 = buf325; del buf325  # reuse
    # Source Nodes: [x_220], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg275_1, reinterpret_tensor(buf336, (1568, 256), (256, 1), 0), reinterpret_tensor(arg274_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf337)
    del arg274_1
    del arg275_1
    buf338 = buf334; del buf334  # reuse
    buf339 = buf333; del buf333  # reuse
    buf341 = reinterpret_tensor(buf331, (8, 768, 196), (150528, 196, 1), 0); del buf331  # reuse
    cpp_fused_clone_native_layer_norm_83(c_void_p(buf337.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()))
    del arg276_1
    del arg277_1
    buf342 = buf330; del buf330  # reuse
    # Source Nodes: [v_83], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (6144, 196), (196, 1), 0), reinterpret_tensor(arg278_1, (196, 196), (1, 196), 0), out=buf342)
    del arg278_1
    buf343 = reinterpret_tensor(buf341, (8, 196, 768), (150528, 768, 1), 0); del buf341  # reuse
    cpp_fused_mul_84(c_void_p(buf337.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(buf343.data_ptr()))
    del arg279_1
    buf344 = reinterpret_tensor(buf336, (1568, 256), (256, 1), 0); del buf336  # reuse
    # Source Nodes: [x_225], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg281_1, reinterpret_tensor(buf343, (1568, 768), (768, 1), 0), reinterpret_tensor(arg280_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf344)
    del arg280_1
    del arg281_1
    buf345 = reinterpret_tensor(buf344, (8, 196, 256), (50176, 256, 1), 0); del buf344  # reuse
    buf346 = buf339; del buf339  # reuse
    buf347 = buf338; del buf338  # reuse
    buf349 = buf247; del buf247  # reuse
    cpp_fused_add_native_layer_norm_85(c_void_p(buf345.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf349.data_ptr()))
    del arg282_1
    del arg283_1
    del buf296
    del buf308
    del buf320
    buf350 = buf337; del buf337  # reuse
    # Source Nodes: [x_228], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg285_1, reinterpret_tensor(buf349, (1568, 256), (256, 1), 0), reinterpret_tensor(arg284_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf350)
    del arg284_1
    del arg285_1
    buf351 = buf347; del buf347  # reuse
    buf352 = buf346; del buf346  # reuse
    buf354 = reinterpret_tensor(buf343, (8, 768, 196), (150528, 196, 1), 0); del buf343  # reuse
    cpp_fused_clone_native_layer_norm_86(c_void_p(buf350.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()))
    del arg286_1
    del arg287_1
    buf355 = buf342; del buf342  # reuse
    # Source Nodes: [v_86], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf354, (6144, 196), (196, 1), 0), reinterpret_tensor(arg288_1, (196, 196), (1, 196), 0), out=buf355)
    del arg288_1
    buf356 = reinterpret_tensor(buf354, (8, 196, 768), (150528, 768, 1), 0); del buf354  # reuse
    cpp_fused_mul_87(c_void_p(buf350.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(buf356.data_ptr()))
    del arg289_1
    buf357 = reinterpret_tensor(buf349, (1568, 256), (256, 1), 0); del buf349  # reuse
    # Source Nodes: [x_233], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg291_1, reinterpret_tensor(buf356, (1568, 768), (768, 1), 0), reinterpret_tensor(arg290_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf357)
    del arg290_1
    del arg291_1
    buf358 = buf352; del buf352  # reuse
    buf359 = buf351; del buf351  # reuse
    buf361 = reinterpret_tensor(buf332, (8, 196, 256), (50176, 256, 1), 0); del buf332  # reuse
    cpp_fused_add_native_layer_norm_88(c_void_p(buf345.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf361.data_ptr()))
    del arg292_1
    del arg293_1
    buf362 = buf350; del buf350  # reuse
    # Source Nodes: [x_236], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg295_1, reinterpret_tensor(buf361, (1568, 256), (256, 1), 0), reinterpret_tensor(arg294_1, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf362)
    del arg294_1
    del arg295_1
    buf363 = buf359; del buf359  # reuse
    buf364 = buf358; del buf358  # reuse
    buf366 = reinterpret_tensor(buf356, (8, 768, 196), (150528, 196, 1), 0); del buf356  # reuse
    cpp_fused_clone_native_layer_norm_89(c_void_p(buf362.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf366.data_ptr()))
    del arg296_1
    del arg297_1
    buf367 = buf355; del buf355  # reuse
    # Source Nodes: [v_89], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf366, (6144, 196), (196, 1), 0), reinterpret_tensor(arg298_1, (196, 196), (1, 196), 0), out=buf367)
    del arg298_1
    buf368 = reinterpret_tensor(buf366, (8, 196, 768), (150528, 768, 1), 0); del buf366  # reuse
    cpp_fused_mul_90(c_void_p(buf362.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(buf368.data_ptr()))
    del arg299_1
    del buf362
    del buf367
    buf369 = reinterpret_tensor(buf361, (1568, 256), (256, 1), 0); del buf361  # reuse
    # Source Nodes: [x_241], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg301_1, reinterpret_tensor(buf368, (1568, 768), (768, 1), 0), reinterpret_tensor(arg300_1, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf369)
    del arg300_1
    del arg301_1
    del buf368
    buf370 = buf364; del buf364  # reuse
    buf371 = buf363; del buf363  # reuse
    buf373 = empty((8, 256), device='cpu', dtype=torch.float32)
    buf374 = buf373; del buf373  # reuse
    cpp_fused_add_mean_native_layer_norm_91(c_void_p(buf374.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    del arg302_1
    del arg303_1
    del buf345
    del buf357
    del buf369
    del buf370
    del buf371
    buf375 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_235, x_244, x_246, x_247, x_249], Original ATen: [aten.add, aten.addmm, aten.mean, aten.native_layer_norm]
    extern_kernels.addmm(arg305_1, buf374, reinterpret_tensor(arg304_1, (256, 1000), (1, 256), 0), alpha=1, beta=1, out=buf375)
    del arg304_1
    del arg305_1
    return (buf375, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((1000, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmlp_s16_224', benchmark_compiled_module)
