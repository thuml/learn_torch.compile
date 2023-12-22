
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
        #pragma omp single
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
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (16L*x1) + (48L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (48L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_native_layer_norm_1 = async_compile.cpp('''
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
                       float* out_ptr6)
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
                        tmp15.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                    auto tmp1 = out_ptr3[static_cast<long>(x0)];
                    auto tmp4 = out_ptr4[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr5 + static_cast<long>(x1 + (64L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (4096L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (64L*x2) + (4096L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
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


cpp_fused_add_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_native_layer_norm_6 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (4096L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (64L*x2) + (4096L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_8 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_10 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (4096L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (64L*x2) + (4096L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_12 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_convolution_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (256L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (256L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_native_layer_norm_15 = async_compile.cpp('''
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
                       float* out_ptr6)
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
                        tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                    auto tmp1 = out_ptr3[static_cast<long>(x0)];
                    auto tmp4 = out_ptr4[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr5 + static_cast<long>(x1 + (128L*x0)));
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
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (128L*x2) + (2048L*x0)), static_cast<long>(128L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_17 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
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


cpp_fused_add_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_native_layer_norm_20 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (2048L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (128L*x2) + (2048L*x0)), static_cast<long>(128L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_24 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (128L*x2) + (2048L*x0)), static_cast<long>(128L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_26 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_28 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (2048L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (128L*x2) + (2048L*x0)), static_cast<long>(128L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_30 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_convolution_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (512L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_native_layer_norm_33 = async_compile.cpp('''
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
                       float* out_ptr6)
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
                        tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                    auto tmp1 = out_ptr3[static_cast<long>(x0)];
                    auto tmp4 = out_ptr4[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr5 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_34 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_native_layer_norm_38 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_39 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_42 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_43 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_44 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_46 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_47 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_48 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_50 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_51 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_52 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_54 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_55 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_56 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_58 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_59 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_60 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_62 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_63 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_66 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_67 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_68 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_70 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_71 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_72 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_74 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_75 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_76 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_78 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_79 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_80 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_82 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_83 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_84 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_86 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_87 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_90 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_91 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_92 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_94 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_95 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_96 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_98 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_99 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_100 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_native_layer_norm_102 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_103 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_104 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2007040L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_convolution_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_107 = async_compile.cpp('''
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
                       float* out_ptr5)
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
                        tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp1 = out_ptr3[static_cast<long>(x0)];
                    auto tmp4 = out_ptr4[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_108 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
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


cpp_fused_add_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_111 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_112 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_114 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_115 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mean_native_layer_norm_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr0;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (25088L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (25088L*x0)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (49L*x0))];
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
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64), (64, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (128, 64), (64, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (64, 64), (64, 1))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (512, 64), (64, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (64, 512), (512, 1))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, 64), (64, 1))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (128, 64), (64, 1))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (64, 64), (64, 1))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (64, ), (1, ))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (512, 64), (64, 1))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (64, 512), (512, 1))
    assert_size_stride(arg41_1, (64, ), (1, ))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, 64), (64, 1))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (64, 64, 8, 8), (4096, 64, 8, 1))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, ), (1, ))
    assert_size_stride(arg50_1, (128, 64), (64, 1))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (64, 64), (64, 1))
    assert_size_stride(arg53_1, (64, ), (1, ))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (512, 64), (64, 1))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (64, 512), (512, 1))
    assert_size_stride(arg59_1, (64, ), (1, ))
    assert_size_stride(arg60_1, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, 128), (128, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (256, 128), (128, 1))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (128, 128), (128, 1))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (1024, 128), (128, 1))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (128, 1024), (1024, 1))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, 128), (128, 1))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (256, 128), (128, 1))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (128, 128), (128, 1))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (128, ), (1, ))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (1024, 128), (128, 1))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (128, 1024), (1024, 1))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, 128), (128, 1))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (256, 128), (128, 1))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (128, 128), (128, 1))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (1024, 128), (128, 1))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (128, 1024), (1024, 1))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, 128), (128, 1))
    assert_size_stride(arg123_1, (128, ), (1, ))
    assert_size_stride(arg124_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (256, 128), (128, 1))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (128, 128), (128, 1))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (1024, 128), (128, 1))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (128, 1024), (1024, 1))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (320, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(arg139_1, (320, ), (1, ))
    assert_size_stride(arg140_1, (320, ), (1, ))
    assert_size_stride(arg141_1, (320, ), (1, ))
    assert_size_stride(arg142_1, (320, ), (1, ))
    assert_size_stride(arg143_1, (320, ), (1, ))
    assert_size_stride(arg144_1, (320, 320), (320, 1))
    assert_size_stride(arg145_1, (320, ), (1, ))
    assert_size_stride(arg146_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg147_1, (320, ), (1, ))
    assert_size_stride(arg148_1, (320, ), (1, ))
    assert_size_stride(arg149_1, (320, ), (1, ))
    assert_size_stride(arg150_1, (640, 320), (320, 1))
    assert_size_stride(arg151_1, (640, ), (1, ))
    assert_size_stride(arg152_1, (320, 320), (320, 1))
    assert_size_stride(arg153_1, (320, ), (1, ))
    assert_size_stride(arg154_1, (320, ), (1, ))
    assert_size_stride(arg155_1, (320, ), (1, ))
    assert_size_stride(arg156_1, (1280, 320), (320, 1))
    assert_size_stride(arg157_1, (1280, ), (1, ))
    assert_size_stride(arg158_1, (320, 1280), (1280, 1))
    assert_size_stride(arg159_1, (320, ), (1, ))
    assert_size_stride(arg160_1, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg161_1, (320, ), (1, ))
    assert_size_stride(arg162_1, (320, ), (1, ))
    assert_size_stride(arg163_1, (320, ), (1, ))
    assert_size_stride(arg164_1, (320, 320), (320, 1))
    assert_size_stride(arg165_1, (320, ), (1, ))
    assert_size_stride(arg166_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg167_1, (320, ), (1, ))
    assert_size_stride(arg168_1, (320, ), (1, ))
    assert_size_stride(arg169_1, (320, ), (1, ))
    assert_size_stride(arg170_1, (640, 320), (320, 1))
    assert_size_stride(arg171_1, (640, ), (1, ))
    assert_size_stride(arg172_1, (320, 320), (320, 1))
    assert_size_stride(arg173_1, (320, ), (1, ))
    assert_size_stride(arg174_1, (320, ), (1, ))
    assert_size_stride(arg175_1, (320, ), (1, ))
    assert_size_stride(arg176_1, (1280, 320), (320, 1))
    assert_size_stride(arg177_1, (1280, ), (1, ))
    assert_size_stride(arg178_1, (320, 1280), (1280, 1))
    assert_size_stride(arg179_1, (320, ), (1, ))
    assert_size_stride(arg180_1, (320, ), (1, ))
    assert_size_stride(arg181_1, (320, ), (1, ))
    assert_size_stride(arg182_1, (320, 320), (320, 1))
    assert_size_stride(arg183_1, (320, ), (1, ))
    assert_size_stride(arg184_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg185_1, (320, ), (1, ))
    assert_size_stride(arg186_1, (320, ), (1, ))
    assert_size_stride(arg187_1, (320, ), (1, ))
    assert_size_stride(arg188_1, (640, 320), (320, 1))
    assert_size_stride(arg189_1, (640, ), (1, ))
    assert_size_stride(arg190_1, (320, 320), (320, 1))
    assert_size_stride(arg191_1, (320, ), (1, ))
    assert_size_stride(arg192_1, (320, ), (1, ))
    assert_size_stride(arg193_1, (320, ), (1, ))
    assert_size_stride(arg194_1, (1280, 320), (320, 1))
    assert_size_stride(arg195_1, (1280, ), (1, ))
    assert_size_stride(arg196_1, (320, 1280), (1280, 1))
    assert_size_stride(arg197_1, (320, ), (1, ))
    assert_size_stride(arg198_1, (320, ), (1, ))
    assert_size_stride(arg199_1, (320, ), (1, ))
    assert_size_stride(arg200_1, (320, 320), (320, 1))
    assert_size_stride(arg201_1, (320, ), (1, ))
    assert_size_stride(arg202_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg203_1, (320, ), (1, ))
    assert_size_stride(arg204_1, (320, ), (1, ))
    assert_size_stride(arg205_1, (320, ), (1, ))
    assert_size_stride(arg206_1, (640, 320), (320, 1))
    assert_size_stride(arg207_1, (640, ), (1, ))
    assert_size_stride(arg208_1, (320, 320), (320, 1))
    assert_size_stride(arg209_1, (320, ), (1, ))
    assert_size_stride(arg210_1, (320, ), (1, ))
    assert_size_stride(arg211_1, (320, ), (1, ))
    assert_size_stride(arg212_1, (1280, 320), (320, 1))
    assert_size_stride(arg213_1, (1280, ), (1, ))
    assert_size_stride(arg214_1, (320, 1280), (1280, 1))
    assert_size_stride(arg215_1, (320, ), (1, ))
    assert_size_stride(arg216_1, (320, ), (1, ))
    assert_size_stride(arg217_1, (320, ), (1, ))
    assert_size_stride(arg218_1, (320, 320), (320, 1))
    assert_size_stride(arg219_1, (320, ), (1, ))
    assert_size_stride(arg220_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg221_1, (320, ), (1, ))
    assert_size_stride(arg222_1, (320, ), (1, ))
    assert_size_stride(arg223_1, (320, ), (1, ))
    assert_size_stride(arg224_1, (640, 320), (320, 1))
    assert_size_stride(arg225_1, (640, ), (1, ))
    assert_size_stride(arg226_1, (320, 320), (320, 1))
    assert_size_stride(arg227_1, (320, ), (1, ))
    assert_size_stride(arg228_1, (320, ), (1, ))
    assert_size_stride(arg229_1, (320, ), (1, ))
    assert_size_stride(arg230_1, (1280, 320), (320, 1))
    assert_size_stride(arg231_1, (1280, ), (1, ))
    assert_size_stride(arg232_1, (320, 1280), (1280, 1))
    assert_size_stride(arg233_1, (320, ), (1, ))
    assert_size_stride(arg234_1, (320, ), (1, ))
    assert_size_stride(arg235_1, (320, ), (1, ))
    assert_size_stride(arg236_1, (320, 320), (320, 1))
    assert_size_stride(arg237_1, (320, ), (1, ))
    assert_size_stride(arg238_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg239_1, (320, ), (1, ))
    assert_size_stride(arg240_1, (320, ), (1, ))
    assert_size_stride(arg241_1, (320, ), (1, ))
    assert_size_stride(arg242_1, (640, 320), (320, 1))
    assert_size_stride(arg243_1, (640, ), (1, ))
    assert_size_stride(arg244_1, (320, 320), (320, 1))
    assert_size_stride(arg245_1, (320, ), (1, ))
    assert_size_stride(arg246_1, (320, ), (1, ))
    assert_size_stride(arg247_1, (320, ), (1, ))
    assert_size_stride(arg248_1, (1280, 320), (320, 1))
    assert_size_stride(arg249_1, (1280, ), (1, ))
    assert_size_stride(arg250_1, (320, 1280), (1280, 1))
    assert_size_stride(arg251_1, (320, ), (1, ))
    assert_size_stride(arg252_1, (320, ), (1, ))
    assert_size_stride(arg253_1, (320, ), (1, ))
    assert_size_stride(arg254_1, (320, 320), (320, 1))
    assert_size_stride(arg255_1, (320, ), (1, ))
    assert_size_stride(arg256_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg257_1, (320, ), (1, ))
    assert_size_stride(arg258_1, (320, ), (1, ))
    assert_size_stride(arg259_1, (320, ), (1, ))
    assert_size_stride(arg260_1, (640, 320), (320, 1))
    assert_size_stride(arg261_1, (640, ), (1, ))
    assert_size_stride(arg262_1, (320, 320), (320, 1))
    assert_size_stride(arg263_1, (320, ), (1, ))
    assert_size_stride(arg264_1, (320, ), (1, ))
    assert_size_stride(arg265_1, (320, ), (1, ))
    assert_size_stride(arg266_1, (1280, 320), (320, 1))
    assert_size_stride(arg267_1, (1280, ), (1, ))
    assert_size_stride(arg268_1, (320, 1280), (1280, 1))
    assert_size_stride(arg269_1, (320, ), (1, ))
    assert_size_stride(arg270_1, (320, ), (1, ))
    assert_size_stride(arg271_1, (320, ), (1, ))
    assert_size_stride(arg272_1, (320, 320), (320, 1))
    assert_size_stride(arg273_1, (320, ), (1, ))
    assert_size_stride(arg274_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg275_1, (320, ), (1, ))
    assert_size_stride(arg276_1, (320, ), (1, ))
    assert_size_stride(arg277_1, (320, ), (1, ))
    assert_size_stride(arg278_1, (640, 320), (320, 1))
    assert_size_stride(arg279_1, (640, ), (1, ))
    assert_size_stride(arg280_1, (320, 320), (320, 1))
    assert_size_stride(arg281_1, (320, ), (1, ))
    assert_size_stride(arg282_1, (320, ), (1, ))
    assert_size_stride(arg283_1, (320, ), (1, ))
    assert_size_stride(arg284_1, (1280, 320), (320, 1))
    assert_size_stride(arg285_1, (1280, ), (1, ))
    assert_size_stride(arg286_1, (320, 1280), (1280, 1))
    assert_size_stride(arg287_1, (320, ), (1, ))
    assert_size_stride(arg288_1, (320, ), (1, ))
    assert_size_stride(arg289_1, (320, ), (1, ))
    assert_size_stride(arg290_1, (320, 320), (320, 1))
    assert_size_stride(arg291_1, (320, ), (1, ))
    assert_size_stride(arg292_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg293_1, (320, ), (1, ))
    assert_size_stride(arg294_1, (320, ), (1, ))
    assert_size_stride(arg295_1, (320, ), (1, ))
    assert_size_stride(arg296_1, (640, 320), (320, 1))
    assert_size_stride(arg297_1, (640, ), (1, ))
    assert_size_stride(arg298_1, (320, 320), (320, 1))
    assert_size_stride(arg299_1, (320, ), (1, ))
    assert_size_stride(arg300_1, (320, ), (1, ))
    assert_size_stride(arg301_1, (320, ), (1, ))
    assert_size_stride(arg302_1, (1280, 320), (320, 1))
    assert_size_stride(arg303_1, (1280, ), (1, ))
    assert_size_stride(arg304_1, (320, 1280), (1280, 1))
    assert_size_stride(arg305_1, (320, ), (1, ))
    assert_size_stride(arg306_1, (320, ), (1, ))
    assert_size_stride(arg307_1, (320, ), (1, ))
    assert_size_stride(arg308_1, (320, 320), (320, 1))
    assert_size_stride(arg309_1, (320, ), (1, ))
    assert_size_stride(arg310_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg311_1, (320, ), (1, ))
    assert_size_stride(arg312_1, (320, ), (1, ))
    assert_size_stride(arg313_1, (320, ), (1, ))
    assert_size_stride(arg314_1, (640, 320), (320, 1))
    assert_size_stride(arg315_1, (640, ), (1, ))
    assert_size_stride(arg316_1, (320, 320), (320, 1))
    assert_size_stride(arg317_1, (320, ), (1, ))
    assert_size_stride(arg318_1, (320, ), (1, ))
    assert_size_stride(arg319_1, (320, ), (1, ))
    assert_size_stride(arg320_1, (1280, 320), (320, 1))
    assert_size_stride(arg321_1, (1280, ), (1, ))
    assert_size_stride(arg322_1, (320, 1280), (1280, 1))
    assert_size_stride(arg323_1, (320, ), (1, ))
    assert_size_stride(arg324_1, (320, ), (1, ))
    assert_size_stride(arg325_1, (320, ), (1, ))
    assert_size_stride(arg326_1, (320, 320), (320, 1))
    assert_size_stride(arg327_1, (320, ), (1, ))
    assert_size_stride(arg328_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg329_1, (320, ), (1, ))
    assert_size_stride(arg330_1, (320, ), (1, ))
    assert_size_stride(arg331_1, (320, ), (1, ))
    assert_size_stride(arg332_1, (640, 320), (320, 1))
    assert_size_stride(arg333_1, (640, ), (1, ))
    assert_size_stride(arg334_1, (320, 320), (320, 1))
    assert_size_stride(arg335_1, (320, ), (1, ))
    assert_size_stride(arg336_1, (320, ), (1, ))
    assert_size_stride(arg337_1, (320, ), (1, ))
    assert_size_stride(arg338_1, (1280, 320), (320, 1))
    assert_size_stride(arg339_1, (1280, ), (1, ))
    assert_size_stride(arg340_1, (320, 1280), (1280, 1))
    assert_size_stride(arg341_1, (320, ), (1, ))
    assert_size_stride(arg342_1, (320, ), (1, ))
    assert_size_stride(arg343_1, (320, ), (1, ))
    assert_size_stride(arg344_1, (320, 320), (320, 1))
    assert_size_stride(arg345_1, (320, ), (1, ))
    assert_size_stride(arg346_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg347_1, (320, ), (1, ))
    assert_size_stride(arg348_1, (320, ), (1, ))
    assert_size_stride(arg349_1, (320, ), (1, ))
    assert_size_stride(arg350_1, (640, 320), (320, 1))
    assert_size_stride(arg351_1, (640, ), (1, ))
    assert_size_stride(arg352_1, (320, 320), (320, 1))
    assert_size_stride(arg353_1, (320, ), (1, ))
    assert_size_stride(arg354_1, (320, ), (1, ))
    assert_size_stride(arg355_1, (320, ), (1, ))
    assert_size_stride(arg356_1, (1280, 320), (320, 1))
    assert_size_stride(arg357_1, (1280, ), (1, ))
    assert_size_stride(arg358_1, (320, 1280), (1280, 1))
    assert_size_stride(arg359_1, (320, ), (1, ))
    assert_size_stride(arg360_1, (320, ), (1, ))
    assert_size_stride(arg361_1, (320, ), (1, ))
    assert_size_stride(arg362_1, (320, 320), (320, 1))
    assert_size_stride(arg363_1, (320, ), (1, ))
    assert_size_stride(arg364_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg365_1, (320, ), (1, ))
    assert_size_stride(arg366_1, (320, ), (1, ))
    assert_size_stride(arg367_1, (320, ), (1, ))
    assert_size_stride(arg368_1, (640, 320), (320, 1))
    assert_size_stride(arg369_1, (640, ), (1, ))
    assert_size_stride(arg370_1, (320, 320), (320, 1))
    assert_size_stride(arg371_1, (320, ), (1, ))
    assert_size_stride(arg372_1, (320, ), (1, ))
    assert_size_stride(arg373_1, (320, ), (1, ))
    assert_size_stride(arg374_1, (1280, 320), (320, 1))
    assert_size_stride(arg375_1, (1280, ), (1, ))
    assert_size_stride(arg376_1, (320, 1280), (1280, 1))
    assert_size_stride(arg377_1, (320, ), (1, ))
    assert_size_stride(arg378_1, (320, ), (1, ))
    assert_size_stride(arg379_1, (320, ), (1, ))
    assert_size_stride(arg380_1, (320, 320), (320, 1))
    assert_size_stride(arg381_1, (320, ), (1, ))
    assert_size_stride(arg382_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg383_1, (320, ), (1, ))
    assert_size_stride(arg384_1, (320, ), (1, ))
    assert_size_stride(arg385_1, (320, ), (1, ))
    assert_size_stride(arg386_1, (640, 320), (320, 1))
    assert_size_stride(arg387_1, (640, ), (1, ))
    assert_size_stride(arg388_1, (320, 320), (320, 1))
    assert_size_stride(arg389_1, (320, ), (1, ))
    assert_size_stride(arg390_1, (320, ), (1, ))
    assert_size_stride(arg391_1, (320, ), (1, ))
    assert_size_stride(arg392_1, (1280, 320), (320, 1))
    assert_size_stride(arg393_1, (1280, ), (1, ))
    assert_size_stride(arg394_1, (320, 1280), (1280, 1))
    assert_size_stride(arg395_1, (320, ), (1, ))
    assert_size_stride(arg396_1, (320, ), (1, ))
    assert_size_stride(arg397_1, (320, ), (1, ))
    assert_size_stride(arg398_1, (320, 320), (320, 1))
    assert_size_stride(arg399_1, (320, ), (1, ))
    assert_size_stride(arg400_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg401_1, (320, ), (1, ))
    assert_size_stride(arg402_1, (320, ), (1, ))
    assert_size_stride(arg403_1, (320, ), (1, ))
    assert_size_stride(arg404_1, (640, 320), (320, 1))
    assert_size_stride(arg405_1, (640, ), (1, ))
    assert_size_stride(arg406_1, (320, 320), (320, 1))
    assert_size_stride(arg407_1, (320, ), (1, ))
    assert_size_stride(arg408_1, (320, ), (1, ))
    assert_size_stride(arg409_1, (320, ), (1, ))
    assert_size_stride(arg410_1, (1280, 320), (320, 1))
    assert_size_stride(arg411_1, (1280, ), (1, ))
    assert_size_stride(arg412_1, (320, 1280), (1280, 1))
    assert_size_stride(arg413_1, (320, ), (1, ))
    assert_size_stride(arg414_1, (320, ), (1, ))
    assert_size_stride(arg415_1, (320, ), (1, ))
    assert_size_stride(arg416_1, (320, 320), (320, 1))
    assert_size_stride(arg417_1, (320, ), (1, ))
    assert_size_stride(arg418_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg419_1, (320, ), (1, ))
    assert_size_stride(arg420_1, (320, ), (1, ))
    assert_size_stride(arg421_1, (320, ), (1, ))
    assert_size_stride(arg422_1, (640, 320), (320, 1))
    assert_size_stride(arg423_1, (640, ), (1, ))
    assert_size_stride(arg424_1, (320, 320), (320, 1))
    assert_size_stride(arg425_1, (320, ), (1, ))
    assert_size_stride(arg426_1, (320, ), (1, ))
    assert_size_stride(arg427_1, (320, ), (1, ))
    assert_size_stride(arg428_1, (1280, 320), (320, 1))
    assert_size_stride(arg429_1, (1280, ), (1, ))
    assert_size_stride(arg430_1, (320, 1280), (1280, 1))
    assert_size_stride(arg431_1, (320, ), (1, ))
    assert_size_stride(arg432_1, (320, ), (1, ))
    assert_size_stride(arg433_1, (320, ), (1, ))
    assert_size_stride(arg434_1, (320, 320), (320, 1))
    assert_size_stride(arg435_1, (320, ), (1, ))
    assert_size_stride(arg436_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg437_1, (320, ), (1, ))
    assert_size_stride(arg438_1, (320, ), (1, ))
    assert_size_stride(arg439_1, (320, ), (1, ))
    assert_size_stride(arg440_1, (640, 320), (320, 1))
    assert_size_stride(arg441_1, (640, ), (1, ))
    assert_size_stride(arg442_1, (320, 320), (320, 1))
    assert_size_stride(arg443_1, (320, ), (1, ))
    assert_size_stride(arg444_1, (320, ), (1, ))
    assert_size_stride(arg445_1, (320, ), (1, ))
    assert_size_stride(arg446_1, (1280, 320), (320, 1))
    assert_size_stride(arg447_1, (1280, ), (1, ))
    assert_size_stride(arg448_1, (320, 1280), (1280, 1))
    assert_size_stride(arg449_1, (320, ), (1, ))
    assert_size_stride(arg450_1, (320, ), (1, ))
    assert_size_stride(arg451_1, (320, ), (1, ))
    assert_size_stride(arg452_1, (320, 320), (320, 1))
    assert_size_stride(arg453_1, (320, ), (1, ))
    assert_size_stride(arg454_1, (320, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg455_1, (320, ), (1, ))
    assert_size_stride(arg456_1, (320, ), (1, ))
    assert_size_stride(arg457_1, (320, ), (1, ))
    assert_size_stride(arg458_1, (640, 320), (320, 1))
    assert_size_stride(arg459_1, (640, ), (1, ))
    assert_size_stride(arg460_1, (320, 320), (320, 1))
    assert_size_stride(arg461_1, (320, ), (1, ))
    assert_size_stride(arg462_1, (320, ), (1, ))
    assert_size_stride(arg463_1, (320, ), (1, ))
    assert_size_stride(arg464_1, (1280, 320), (320, 1))
    assert_size_stride(arg465_1, (1280, ), (1, ))
    assert_size_stride(arg466_1, (320, 1280), (1280, 1))
    assert_size_stride(arg467_1, (320, ), (1, ))
    assert_size_stride(arg468_1, (512, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg469_1, (512, ), (1, ))
    assert_size_stride(arg470_1, (512, ), (1, ))
    assert_size_stride(arg471_1, (512, ), (1, ))
    assert_size_stride(arg472_1, (512, ), (1, ))
    assert_size_stride(arg473_1, (512, ), (1, ))
    assert_size_stride(arg474_1, (512, 512), (512, 1))
    assert_size_stride(arg475_1, (512, ), (1, ))
    assert_size_stride(arg476_1, (1024, 512), (512, 1))
    assert_size_stride(arg477_1, (1024, ), (1, ))
    assert_size_stride(arg478_1, (512, 512), (512, 1))
    assert_size_stride(arg479_1, (512, ), (1, ))
    assert_size_stride(arg480_1, (512, ), (1, ))
    assert_size_stride(arg481_1, (512, ), (1, ))
    assert_size_stride(arg482_1, (2048, 512), (512, 1))
    assert_size_stride(arg483_1, (2048, ), (1, ))
    assert_size_stride(arg484_1, (512, 2048), (2048, 1))
    assert_size_stride(arg485_1, (512, ), (1, ))
    assert_size_stride(arg486_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg487_1, (512, ), (1, ))
    assert_size_stride(arg488_1, (512, ), (1, ))
    assert_size_stride(arg489_1, (512, ), (1, ))
    assert_size_stride(arg490_1, (512, 512), (512, 1))
    assert_size_stride(arg491_1, (512, ), (1, ))
    assert_size_stride(arg492_1, (1024, 512), (512, 1))
    assert_size_stride(arg493_1, (1024, ), (1, ))
    assert_size_stride(arg494_1, (512, 512), (512, 1))
    assert_size_stride(arg495_1, (512, ), (1, ))
    assert_size_stride(arg496_1, (512, ), (1, ))
    assert_size_stride(arg497_1, (512, ), (1, ))
    assert_size_stride(arg498_1, (2048, 512), (512, 1))
    assert_size_stride(arg499_1, (2048, ), (1, ))
    assert_size_stride(arg500_1, (512, 2048), (2048, 1))
    assert_size_stride(arg501_1, (512, ), (1, ))
    assert_size_stride(arg502_1, (512, ), (1, ))
    assert_size_stride(arg503_1, (512, ), (1, ))
    assert_size_stride(arg504_1, (512, 512), (512, 1))
    assert_size_stride(arg505_1, (512, ), (1, ))
    assert_size_stride(arg506_1, (1024, 512), (512, 1))
    assert_size_stride(arg507_1, (1024, ), (1, ))
    assert_size_stride(arg508_1, (512, 512), (512, 1))
    assert_size_stride(arg509_1, (512, ), (1, ))
    assert_size_stride(arg510_1, (512, ), (1, ))
    assert_size_stride(arg511_1, (512, ), (1, ))
    assert_size_stride(arg512_1, (2048, 512), (512, 1))
    assert_size_stride(arg513_1, (2048, ), (1, ))
    assert_size_stride(arg514_1, (512, 2048), (2048, 1))
    assert_size_stride(arg515_1, (512, ), (1, ))
    assert_size_stride(arg516_1, (512, ), (1, ))
    assert_size_stride(arg517_1, (512, ), (1, ))
    assert_size_stride(arg518_1, (1000, 512), (512, 1))
    assert_size_stride(arg519_1, (1000, ), (1, ))
    assert_size_stride(arg520_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg520_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg520_1
    # Source Nodes: [l__mod___patch_embeds_0_proj], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg1_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del arg1_1
    del buf0
    del buf1
    buf3 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 3136, 64), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf10 = empty((8, 3136, 64), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    del arg2_1
    del arg3_1
    del arg4_1
    del arg5_1
    del arg8_1
    del buf3
    del buf4
    del buf7
    # Source Nodes: [l__mod___blocks_0_0_attn_sr], Original ATen: [aten.convolution]
    buf12 = extern_kernels.convolution(reinterpret_tensor(buf10, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), buf11, arg9_1, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf12, (8, 64, 7, 7), (3136, 1, 448, 64))
    del arg9_1
    buf13 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf16 = reinterpret_tensor(buf8, (8, 49, 64), (3136, 64, 1), 0); del buf8  # reuse
    cpp_fused_native_layer_norm_2(c_void_p(buf12.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()))
    del arg10_1
    del arg11_1
    buf17 = empty((392, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_0_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg13_1, reinterpret_tensor(buf16, (392, 64), (64, 1), 0), reinterpret_tensor(arg12_1, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf17)
    del arg12_1
    del arg13_1
    buf18 = reinterpret_tensor(buf2, (25088, 64), (64, 1), 0); del buf2  # reuse
    # Source Nodes: [l__mod___blocks_0_0_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg7_1, reinterpret_tensor(buf10, (25088, 64), (64, 1), 0), reinterpret_tensor(arg6_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf18)
    del arg6_1
    del arg7_1
    # Source Nodes: [x_7], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf19 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf18, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf17, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf17, (8, 1, 49, 64), (6272, 0, 128, 1), 64))
    buf20 = buf19[0]
    del buf19
    buf27 = buf18; del buf18  # reuse
    # Source Nodes: [x_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg15_1, reinterpret_tensor(buf20, (25088, 64), (64, 1), 0), reinterpret_tensor(arg14_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf27)
    del arg14_1
    del arg15_1
    buf28 = reinterpret_tensor(buf16, (8, 3136, 1), (3136, 1, 25088), 0); del buf16  # reuse
    buf29 = reinterpret_tensor(buf12, (8, 3136, 1), (3136, 1, 25088), 0); del buf12  # reuse
    buf31 = reinterpret_tensor(buf20, (8, 3136, 64), (200704, 64, 1), 0); del buf20  # reuse
    cpp_fused_add_native_layer_norm_3(c_void_p(buf6.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg16_1
    del arg17_1
    buf32 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg19_1, reinterpret_tensor(buf31, (25088, 64), (64, 1), 0), reinterpret_tensor(arg18_1, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf32)
    del arg18_1
    del arg19_1
    buf33 = reinterpret_tensor(buf32, (8, 3136, 512), (1605632, 512, 1), 0); del buf32  # reuse
    cpp_fused_gelu_4(c_void_p(buf33.data_ptr()))
    buf34 = reinterpret_tensor(buf31, (25088, 64), (64, 1), 0); del buf31  # reuse
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg21_1, reinterpret_tensor(buf33, (25088, 512), (512, 1), 0), reinterpret_tensor(arg20_1, (512, 64), (1, 512), 0), alpha=1, beta=1, out=buf34)
    del arg20_1
    del arg21_1
    buf35 = reinterpret_tensor(buf34, (8, 3136, 64), (200704, 64, 1), 0); del buf34  # reuse
    cpp_fused_add_5(c_void_p(buf35.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf27.data_ptr()))
    # Source Nodes: [x_20], Original ATen: [aten.convolution]
    buf36 = extern_kernels.convolution(reinterpret_tensor(buf35, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), arg22_1, arg23_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64)
    assert_size_stride(buf36, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del arg22_1
    del arg23_1
    buf37 = buf29; del buf29  # reuse
    buf38 = buf28; del buf28  # reuse
    buf40 = buf6; del buf6  # reuse
    buf41 = buf11; del buf11  # reuse
    cpp_fused_convolution_native_layer_norm_6(c_void_p(buf36.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    del arg24_1
    del arg25_1
    del arg28_1
    del buf37
    # Source Nodes: [l__mod___blocks_0_1_attn_sr], Original ATen: [aten.convolution]
    buf42 = extern_kernels.convolution(reinterpret_tensor(buf40, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), buf41, arg29_1, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf42, (8, 64, 7, 7), (3136, 1, 448, 64))
    del arg29_1
    buf43 = buf14; del buf14  # reuse
    buf44 = buf13; del buf13  # reuse
    buf46 = reinterpret_tensor(buf38, (8, 49, 64), (3136, 64, 1), 0); del buf38  # reuse
    cpp_fused_native_layer_norm_7(c_void_p(buf42.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()))
    del arg30_1
    del arg31_1
    buf47 = buf17; del buf17  # reuse
    # Source Nodes: [l__mod___blocks_0_1_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg33_1, reinterpret_tensor(buf46, (392, 64), (64, 1), 0), reinterpret_tensor(arg32_1, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf47)
    del arg32_1
    del arg33_1
    buf48 = buf27; del buf27  # reuse
    # Source Nodes: [l__mod___blocks_0_1_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg27_1, reinterpret_tensor(buf40, (25088, 64), (64, 1), 0), reinterpret_tensor(arg26_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf48)
    del arg26_1
    del arg27_1
    # Source Nodes: [x_27], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf49 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf48, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf47, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf47, (8, 1, 49, 64), (6272, 0, 128, 1), 64))
    buf50 = buf49[0]
    del buf49
    buf57 = buf48; del buf48  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg35_1, reinterpret_tensor(buf50, (25088, 64), (64, 1), 0), reinterpret_tensor(arg34_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf57)
    del arg34_1
    del arg35_1
    buf58 = reinterpret_tensor(buf46, (8, 3136, 1), (3136, 1, 25088), 0); del buf46  # reuse
    buf59 = reinterpret_tensor(buf42, (8, 3136, 1), (3136, 1, 25088), 0); del buf42  # reuse
    buf61 = reinterpret_tensor(buf50, (8, 3136, 64), (200704, 64, 1), 0); del buf50  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf36.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()))
    del arg36_1
    del arg37_1
    buf62 = reinterpret_tensor(buf33, (25088, 512), (512, 1), 0); del buf33  # reuse
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg39_1, reinterpret_tensor(buf61, (25088, 64), (64, 1), 0), reinterpret_tensor(arg38_1, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf62)
    del arg38_1
    del arg39_1
    buf63 = reinterpret_tensor(buf62, (8, 3136, 512), (1605632, 512, 1), 0); del buf62  # reuse
    cpp_fused_gelu_9(c_void_p(buf63.data_ptr()))
    buf64 = reinterpret_tensor(buf61, (25088, 64), (64, 1), 0); del buf61  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg41_1, reinterpret_tensor(buf63, (25088, 512), (512, 1), 0), reinterpret_tensor(arg40_1, (512, 64), (1, 512), 0), alpha=1, beta=1, out=buf64)
    del arg40_1
    del arg41_1
    buf65 = buf59; del buf59  # reuse
    buf66 = buf58; del buf58  # reuse
    buf68 = buf40; del buf40  # reuse
    buf69 = buf41; del buf41  # reuse
    cpp_fused_add_convolution_native_layer_norm_10(c_void_p(buf36.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()))
    del arg42_1
    del arg43_1
    del arg46_1
    del buf65
    # Source Nodes: [l__mod___blocks_0_2_attn_sr], Original ATen: [aten.convolution]
    buf70 = extern_kernels.convolution(reinterpret_tensor(buf68, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), buf69, arg47_1, stride=(8, 8), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf70, (8, 64, 7, 7), (3136, 1, 448, 64))
    del arg47_1
    buf71 = buf44; del buf44  # reuse
    buf72 = buf43; del buf43  # reuse
    buf74 = reinterpret_tensor(buf66, (8, 49, 64), (3136, 64, 1), 0); del buf66  # reuse
    cpp_fused_native_layer_norm_11(c_void_p(buf70.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()))
    del arg48_1
    del arg49_1
    buf75 = buf47; del buf47  # reuse
    # Source Nodes: [l__mod___blocks_0_2_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg51_1, reinterpret_tensor(buf74, (392, 64), (64, 1), 0), reinterpret_tensor(arg50_1, (64, 128), (1, 64), 0), alpha=1, beta=1, out=buf75)
    del arg50_1
    del arg51_1
    buf76 = reinterpret_tensor(buf10, (25088, 64), (64, 1), 0); del buf10  # reuse
    # Source Nodes: [l__mod___blocks_0_2_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg45_1, reinterpret_tensor(buf68, (25088, 64), (64, 1), 0), reinterpret_tensor(arg44_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf76)
    del arg44_1
    del arg45_1
    del buf68
    # Source Nodes: [x_43], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf77 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf76, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), reinterpret_tensor(buf75, (8, 1, 49, 64), (6272, 0, 128, 1), 0), reinterpret_tensor(buf75, (8, 1, 49, 64), (6272, 0, 128, 1), 64))
    buf78 = buf77[0]
    del buf77
    buf85 = buf76; del buf76  # reuse
    # Source Nodes: [x_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg53_1, reinterpret_tensor(buf78, (25088, 64), (64, 1), 0), reinterpret_tensor(arg52_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf85)
    del arg52_1
    del arg53_1
    buf86 = reinterpret_tensor(buf85, (8, 3136, 64), (200704, 64, 1), 0); del buf85  # reuse
    buf87 = reinterpret_tensor(buf74, (8, 3136, 1), (3136, 1, 25088), 0); del buf74  # reuse
    buf88 = reinterpret_tensor(buf70, (8, 3136, 1), (3136, 1, 25088), 0); del buf70  # reuse
    buf90 = reinterpret_tensor(buf78, (8, 3136, 64), (200704, 64, 1), 0); del buf78  # reuse
    cpp_fused_add_native_layer_norm_12(c_void_p(buf86.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()))
    del arg54_1
    del arg55_1
    del buf35
    del buf36
    del buf57
    del buf64
    del buf87
    del buf88
    buf91 = reinterpret_tensor(buf63, (25088, 512), (512, 1), 0); del buf63  # reuse
    # Source Nodes: [x_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg57_1, reinterpret_tensor(buf90, (25088, 64), (64, 1), 0), reinterpret_tensor(arg56_1, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf91)
    del arg56_1
    del arg57_1
    buf92 = reinterpret_tensor(buf91, (8, 3136, 512), (1605632, 512, 1), 0); del buf91  # reuse
    cpp_fused_gelu_13(c_void_p(buf92.data_ptr()))
    buf93 = reinterpret_tensor(buf90, (25088, 64), (64, 1), 0); del buf90  # reuse
    # Source Nodes: [x_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg59_1, reinterpret_tensor(buf92, (25088, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 64), (1, 512), 0), alpha=1, beta=1, out=buf93)
    del arg58_1
    del arg59_1
    del buf92
    buf94 = reinterpret_tensor(buf93, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf93  # reuse
    buf95 = empty_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_convolution_14(c_void_p(buf94.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf95.data_ptr()))
    del arg60_1
    del buf86
    # Source Nodes: [l__mod___patch_embeds_1_proj, x_56], Original ATen: [aten.clone, aten.convolution]
    buf96 = extern_kernels.convolution(buf94, buf95, arg61_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf96, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del arg61_1
    del buf94
    del buf95
    buf97 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf98 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf100 = empty((8, 784, 128), device='cpu', dtype=torch.float32)
    buf101 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf102 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf104 = empty((8, 784, 128), device='cpu', dtype=torch.float32)
    buf105 = reinterpret_tensor(buf69, (128, 128, 4, 4), (2048, 1, 512, 128), 0); del buf69  # reuse
    cpp_fused_convolution_native_layer_norm_15(c_void_p(buf96.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del arg62_1
    del arg63_1
    del arg64_1
    del arg65_1
    del arg68_1
    del buf101
    del buf102
    # Source Nodes: [l__mod___blocks_1_0_attn_sr], Original ATen: [aten.convolution]
    buf106 = extern_kernels.convolution(reinterpret_tensor(buf104, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf105, arg69_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf106, (8, 128, 7, 7), (6272, 1, 896, 128))
    del arg69_1
    buf107 = buf72; del buf72  # reuse
    buf108 = buf71; del buf71  # reuse
    buf110 = reinterpret_tensor(buf75, (8, 49, 128), (6272, 128, 1), 0); del buf75  # reuse
    cpp_fused_native_layer_norm_16(c_void_p(buf106.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()))
    del arg70_1
    del arg71_1
    del buf106
    buf111 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_0_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf110, (392, 128), (128, 1), 0), reinterpret_tensor(arg72_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf111)
    del arg72_1
    del arg73_1
    buf112 = reinterpret_tensor(buf96, (6272, 128), (128, 1), 0); del buf96  # reuse
    # Source Nodes: [l__mod___blocks_1_0_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg67_1, reinterpret_tensor(buf104, (6272, 128), (128, 1), 0), reinterpret_tensor(arg66_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf112)
    del arg66_1
    del arg67_1
    # Source Nodes: [x_64], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf113 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf112, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf111, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf111, (8, 2, 49, 64), (12544, 64, 256, 1), 128))
    buf114 = buf113[0]
    del buf113
    buf121 = buf112; del buf112  # reuse
    # Source Nodes: [x_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg75_1, reinterpret_tensor(buf114, (6272, 128), (128, 1), 0), reinterpret_tensor(arg74_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf121)
    del arg74_1
    del arg75_1
    buf122 = buf98; del buf98  # reuse
    buf123 = buf97; del buf97  # reuse
    buf125 = reinterpret_tensor(buf114, (8, 784, 128), (100352, 128, 1), 0); del buf114  # reuse
    cpp_fused_add_native_layer_norm_17(c_void_p(buf100.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()))
    del arg76_1
    del arg77_1
    buf126 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg79_1, reinterpret_tensor(buf125, (6272, 128), (128, 1), 0), reinterpret_tensor(arg78_1, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf126)
    del arg78_1
    del arg79_1
    buf127 = reinterpret_tensor(buf126, (8, 784, 1024), (802816, 1024, 1), 0); del buf126  # reuse
    cpp_fused_gelu_18(c_void_p(buf127.data_ptr()))
    buf128 = reinterpret_tensor(buf125, (6272, 128), (128, 1), 0); del buf125  # reuse
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg81_1, reinterpret_tensor(buf127, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg80_1, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf128)
    del arg80_1
    del arg81_1
    buf129 = reinterpret_tensor(buf128, (8, 784, 128), (100352, 128, 1), 0); del buf128  # reuse
    cpp_fused_add_19(c_void_p(buf129.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf121.data_ptr()))
    # Source Nodes: [x_77], Original ATen: [aten.convolution]
    buf130 = extern_kernels.convolution(reinterpret_tensor(buf129, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), arg82_1, arg83_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf130, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del arg82_1
    del arg83_1
    buf131 = buf123; del buf123  # reuse
    buf132 = buf122; del buf122  # reuse
    buf134 = reinterpret_tensor(buf121, (8, 784, 128), (100352, 128, 1), 0); del buf121  # reuse
    buf135 = buf105; del buf105  # reuse
    cpp_fused_convolution_native_layer_norm_20(c_void_p(buf130.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg84_1
    del arg85_1
    del arg88_1
    # Source Nodes: [l__mod___blocks_1_1_attn_sr], Original ATen: [aten.convolution]
    buf136 = extern_kernels.convolution(reinterpret_tensor(buf134, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf135, arg89_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf136, (8, 128, 7, 7), (6272, 1, 896, 128))
    del arg89_1
    buf137 = buf108; del buf108  # reuse
    buf138 = buf107; del buf107  # reuse
    buf140 = buf110; del buf110  # reuse
    cpp_fused_native_layer_norm_21(c_void_p(buf136.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()))
    del arg90_1
    del arg91_1
    del buf136
    buf141 = buf111; del buf111  # reuse
    # Source Nodes: [l__mod___blocks_1_1_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg93_1, reinterpret_tensor(buf140, (392, 128), (128, 1), 0), reinterpret_tensor(arg92_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf141)
    del arg92_1
    del arg93_1
    buf142 = reinterpret_tensor(buf100, (6272, 128), (128, 1), 0); del buf100  # reuse
    # Source Nodes: [l__mod___blocks_1_1_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf134, (6272, 128), (128, 1), 0), reinterpret_tensor(arg86_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf142)
    del arg86_1
    del arg87_1
    # Source Nodes: [x_84], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf143 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf142, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf141, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf141, (8, 2, 49, 64), (12544, 64, 256, 1), 128))
    buf144 = buf143[0]
    del buf143
    buf151 = buf142; del buf142  # reuse
    # Source Nodes: [x_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, reinterpret_tensor(buf144, (6272, 128), (128, 1), 0), reinterpret_tensor(arg94_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf151)
    del arg94_1
    del arg95_1
    buf152 = buf132; del buf132  # reuse
    buf153 = buf131; del buf131  # reuse
    buf155 = reinterpret_tensor(buf144, (8, 784, 128), (100352, 128, 1), 0); del buf144  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf130.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf155.data_ptr()))
    del arg96_1
    del arg97_1
    buf156 = reinterpret_tensor(buf127, (6272, 1024), (1024, 1), 0); del buf127  # reuse
    # Source Nodes: [x_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg99_1, reinterpret_tensor(buf155, (6272, 128), (128, 1), 0), reinterpret_tensor(arg98_1, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf156)
    del arg98_1
    del arg99_1
    buf157 = reinterpret_tensor(buf156, (8, 784, 1024), (802816, 1024, 1), 0); del buf156  # reuse
    cpp_fused_gelu_23(c_void_p(buf157.data_ptr()))
    buf158 = reinterpret_tensor(buf155, (6272, 128), (128, 1), 0); del buf155  # reuse
    # Source Nodes: [x_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf157, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg100_1, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf158)
    del arg100_1
    del arg101_1
    buf159 = buf153; del buf153  # reuse
    buf160 = buf152; del buf152  # reuse
    buf162 = buf134; del buf134  # reuse
    buf163 = buf135; del buf135  # reuse
    cpp_fused_add_convolution_native_layer_norm_24(c_void_p(buf130.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    del arg102_1
    del arg103_1
    del arg106_1
    # Source Nodes: [l__mod___blocks_1_2_attn_sr], Original ATen: [aten.convolution]
    buf164 = extern_kernels.convolution(reinterpret_tensor(buf162, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf163, arg107_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf164, (8, 128, 7, 7), (6272, 1, 896, 128))
    del arg107_1
    buf165 = buf138; del buf138  # reuse
    buf166 = buf137; del buf137  # reuse
    buf168 = buf140; del buf140  # reuse
    cpp_fused_native_layer_norm_25(c_void_p(buf164.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()))
    del arg108_1
    del arg109_1
    del buf164
    buf169 = buf141; del buf141  # reuse
    # Source Nodes: [l__mod___blocks_1_2_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg111_1, reinterpret_tensor(buf168, (392, 128), (128, 1), 0), reinterpret_tensor(arg110_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf169)
    del arg110_1
    del arg111_1
    buf170 = reinterpret_tensor(buf104, (6272, 128), (128, 1), 0); del buf104  # reuse
    # Source Nodes: [l__mod___blocks_1_2_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf162, (6272, 128), (128, 1), 0), reinterpret_tensor(arg104_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf170)
    del arg104_1
    del arg105_1
    del buf162
    # Source Nodes: [x_100], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf171 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf170, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf169, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf169, (8, 2, 49, 64), (12544, 64, 256, 1), 128))
    buf172 = buf171[0]
    del buf171
    buf179 = buf170; del buf170  # reuse
    # Source Nodes: [x_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg113_1, reinterpret_tensor(buf172, (6272, 128), (128, 1), 0), reinterpret_tensor(arg112_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf179)
    del arg112_1
    del arg113_1
    buf180 = reinterpret_tensor(buf179, (8, 784, 128), (100352, 128, 1), 0); del buf179  # reuse
    buf181 = buf160; del buf160  # reuse
    buf182 = buf159; del buf159  # reuse
    buf184 = reinterpret_tensor(buf172, (8, 784, 128), (100352, 128, 1), 0); del buf172  # reuse
    cpp_fused_add_native_layer_norm_26(c_void_p(buf180.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()))
    del arg114_1
    del arg115_1
    del buf129
    del buf130
    buf185 = reinterpret_tensor(buf157, (6272, 1024), (1024, 1), 0); del buf157  # reuse
    # Source Nodes: [x_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg117_1, reinterpret_tensor(buf184, (6272, 128), (128, 1), 0), reinterpret_tensor(arg116_1, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf185)
    del arg116_1
    del arg117_1
    buf186 = reinterpret_tensor(buf185, (8, 784, 1024), (802816, 1024, 1), 0); del buf185  # reuse
    cpp_fused_gelu_27(c_void_p(buf186.data_ptr()))
    buf187 = reinterpret_tensor(buf184, (6272, 128), (128, 1), 0); del buf184  # reuse
    # Source Nodes: [x_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf186, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf187)
    del arg118_1
    del arg119_1
    buf188 = buf182; del buf182  # reuse
    buf189 = buf181; del buf181  # reuse
    buf191 = reinterpret_tensor(buf158, (8, 784, 128), (100352, 128, 1), 0); del buf158  # reuse
    buf192 = buf163; del buf163  # reuse
    cpp_fused_add_convolution_native_layer_norm_28(c_void_p(buf180.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    del arg120_1
    del arg121_1
    del arg124_1
    # Source Nodes: [l__mod___blocks_1_3_attn_sr], Original ATen: [aten.convolution]
    buf193 = extern_kernels.convolution(reinterpret_tensor(buf191, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), buf192, arg125_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf193, (8, 128, 7, 7), (6272, 1, 896, 128))
    del arg125_1
    del buf192
    buf194 = buf166; del buf166  # reuse
    buf195 = buf165; del buf165  # reuse
    buf197 = buf168; del buf168  # reuse
    cpp_fused_native_layer_norm_29(c_void_p(buf193.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf197.data_ptr()))
    del arg126_1
    del arg127_1
    del buf193
    buf198 = buf169; del buf169  # reuse
    # Source Nodes: [l__mod___blocks_1_3_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf197, (392, 128), (128, 1), 0), reinterpret_tensor(arg128_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf198)
    del arg128_1
    del arg129_1
    del buf197
    buf199 = buf151; del buf151  # reuse
    # Source Nodes: [l__mod___blocks_1_3_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg123_1, reinterpret_tensor(buf191, (6272, 128), (128, 1), 0), reinterpret_tensor(arg122_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf199)
    del arg122_1
    del arg123_1
    del buf191
    # Source Nodes: [x_116], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf200 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf199, (8, 2, 784, 64), (100352, 64, 128, 1), 0), reinterpret_tensor(buf198, (8, 2, 49, 64), (12544, 64, 256, 1), 0), reinterpret_tensor(buf198, (8, 2, 49, 64), (12544, 64, 256, 1), 128))
    del buf198
    buf201 = buf200[0]
    del buf200
    buf208 = buf199; del buf199  # reuse
    # Source Nodes: [x_118], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf201, (6272, 128), (128, 1), 0), reinterpret_tensor(arg130_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf208)
    del arg130_1
    del arg131_1
    buf209 = buf189; del buf189  # reuse
    buf210 = buf188; del buf188  # reuse
    buf212 = reinterpret_tensor(buf201, (8, 784, 128), (100352, 128, 1), 0); del buf201  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf180.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()))
    del arg132_1
    del arg133_1
    del buf209
    del buf210
    buf213 = reinterpret_tensor(buf186, (6272, 1024), (1024, 1), 0); del buf186  # reuse
    # Source Nodes: [x_121], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg135_1, reinterpret_tensor(buf212, (6272, 128), (128, 1), 0), reinterpret_tensor(arg134_1, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf213)
    del arg134_1
    del arg135_1
    buf214 = reinterpret_tensor(buf213, (8, 784, 1024), (802816, 1024, 1), 0); del buf213  # reuse
    cpp_fused_gelu_31(c_void_p(buf214.data_ptr()))
    buf215 = reinterpret_tensor(buf212, (6272, 128), (128, 1), 0); del buf212  # reuse
    # Source Nodes: [x_125], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg137_1, reinterpret_tensor(buf214, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf215)
    del arg136_1
    del arg137_1
    del buf214
    buf216 = reinterpret_tensor(buf215, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf215  # reuse
    buf217 = empty_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_convolution_32(c_void_p(buf216.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(buf217.data_ptr()))
    del arg138_1
    del buf180
    del buf187
    del buf208
    # Source Nodes: [l__mod___patch_embeds_2_proj, x_129], Original ATen: [aten.clone, aten.convolution]
    buf218 = extern_kernels.convolution(buf216, buf217, arg139_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf218, (8, 320, 14, 14), (62720, 1, 4480, 320))
    del arg139_1
    del buf217
    buf219 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf220 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf222 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf223 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf224 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf226 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf227 = empty_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_native_layer_norm_33(c_void_p(buf218.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()))
    del arg140_1
    del arg141_1
    del arg142_1
    del arg143_1
    del arg146_1
    del buf219
    del buf220
    # Source Nodes: [l__mod___blocks_2_0_attn_sr], Original ATen: [aten.convolution]
    buf228 = extern_kernels.convolution(reinterpret_tensor(buf226, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf227, arg147_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf228, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg147_1
    buf229 = buf195; del buf195  # reuse
    buf230 = buf194; del buf194  # reuse
    buf232 = empty((8, 49, 320), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_34(c_void_p(buf228.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()))
    del arg148_1
    del arg149_1
    del buf228
    buf233 = empty((392, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_0_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg151_1, reinterpret_tensor(buf232, (392, 320), (320, 1), 0), reinterpret_tensor(arg150_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf233)
    del arg150_1
    del arg151_1
    buf234 = reinterpret_tensor(buf218, (1568, 320), (320, 1), 0); del buf218  # reuse
    # Source Nodes: [l__mod___blocks_2_0_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf226, (1568, 320), (320, 1), 0), reinterpret_tensor(arg144_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf234)
    del arg144_1
    del arg145_1
    # Source Nodes: [x_137], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf235 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf234, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf233, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf233, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf236 = buf235[0]
    del buf235
    buf243 = buf234; del buf234  # reuse
    # Source Nodes: [x_139], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg153_1, reinterpret_tensor(buf236, (1568, 320), (320, 1), 0), reinterpret_tensor(arg152_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf243)
    del arg152_1
    del arg153_1
    buf244 = buf224; del buf224  # reuse
    buf245 = buf223; del buf223  # reuse
    buf247 = reinterpret_tensor(buf236, (8, 196, 320), (62720, 320, 1), 0); del buf236  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf222.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()))
    del arg154_1
    del arg155_1
    buf248 = empty((1568, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg157_1, reinterpret_tensor(buf247, (1568, 320), (320, 1), 0), reinterpret_tensor(arg156_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf248)
    del arg156_1
    del arg157_1
    buf249 = reinterpret_tensor(buf248, (8, 196, 1280), (250880, 1280, 1), 0); del buf248  # reuse
    cpp_fused_gelu_36(c_void_p(buf249.data_ptr()))
    buf250 = reinterpret_tensor(buf247, (1568, 320), (320, 1), 0); del buf247  # reuse
    # Source Nodes: [x_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg159_1, reinterpret_tensor(buf249, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg158_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf250)
    del arg158_1
    del arg159_1
    buf251 = reinterpret_tensor(buf250, (8, 196, 320), (62720, 320, 1), 0); del buf250  # reuse
    cpp_fused_add_37(c_void_p(buf251.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf243.data_ptr()))
    # Source Nodes: [x_150], Original ATen: [aten.convolution]
    buf252 = extern_kernels.convolution(reinterpret_tensor(buf251, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), arg160_1, arg161_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320)
    assert_size_stride(buf252, (8, 320, 14, 14), (62720, 1, 4480, 320))
    del arg160_1
    del arg161_1
    buf253 = buf245; del buf245  # reuse
    buf254 = buf244; del buf244  # reuse
    buf256 = reinterpret_tensor(buf243, (8, 196, 320), (62720, 320, 1), 0); del buf243  # reuse
    buf257 = buf227; del buf227  # reuse
    cpp_fused_convolution_native_layer_norm_38(c_void_p(buf252.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()))
    del arg162_1
    del arg163_1
    del arg166_1
    # Source Nodes: [l__mod___blocks_2_1_attn_sr], Original ATen: [aten.convolution]
    buf258 = extern_kernels.convolution(reinterpret_tensor(buf256, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf257, arg167_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf258, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg167_1
    buf259 = buf230; del buf230  # reuse
    buf260 = buf229; del buf229  # reuse
    buf262 = buf232; del buf232  # reuse
    cpp_fused_native_layer_norm_39(c_void_p(buf258.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf262.data_ptr()))
    del arg168_1
    del arg169_1
    del buf258
    buf263 = buf233; del buf233  # reuse
    # Source Nodes: [l__mod___blocks_2_1_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg171_1, reinterpret_tensor(buf262, (392, 320), (320, 1), 0), reinterpret_tensor(arg170_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf263)
    del arg170_1
    del arg171_1
    buf264 = reinterpret_tensor(buf222, (1568, 320), (320, 1), 0); del buf222  # reuse
    # Source Nodes: [l__mod___blocks_2_1_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg165_1, reinterpret_tensor(buf256, (1568, 320), (320, 1), 0), reinterpret_tensor(arg164_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf264)
    del arg164_1
    del arg165_1
    # Source Nodes: [x_157], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf265 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf264, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf263, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf263, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf266 = buf265[0]
    del buf265
    buf273 = buf264; del buf264  # reuse
    # Source Nodes: [x_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg173_1, reinterpret_tensor(buf266, (1568, 320), (320, 1), 0), reinterpret_tensor(arg172_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf273)
    del arg172_1
    del arg173_1
    buf274 = buf254; del buf254  # reuse
    buf275 = buf253; del buf253  # reuse
    buf277 = reinterpret_tensor(buf266, (8, 196, 320), (62720, 320, 1), 0); del buf266  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf252.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf277.data_ptr()))
    del arg174_1
    del arg175_1
    buf278 = reinterpret_tensor(buf249, (1568, 1280), (1280, 1), 0); del buf249  # reuse
    # Source Nodes: [x_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg177_1, reinterpret_tensor(buf277, (1568, 320), (320, 1), 0), reinterpret_tensor(arg176_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf278)
    del arg176_1
    del arg177_1
    buf279 = reinterpret_tensor(buf278, (8, 196, 1280), (250880, 1280, 1), 0); del buf278  # reuse
    cpp_fused_gelu_41(c_void_p(buf279.data_ptr()))
    buf280 = reinterpret_tensor(buf277, (1568, 320), (320, 1), 0); del buf277  # reuse
    # Source Nodes: [x_166], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg179_1, reinterpret_tensor(buf279, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg178_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf280)
    del arg178_1
    del arg179_1
    buf281 = buf275; del buf275  # reuse
    buf282 = buf274; del buf274  # reuse
    buf284 = buf256; del buf256  # reuse
    buf285 = buf257; del buf257  # reuse
    cpp_fused_add_convolution_native_layer_norm_42(c_void_p(buf252.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    del arg180_1
    del arg181_1
    del arg184_1
    # Source Nodes: [l__mod___blocks_2_2_attn_sr], Original ATen: [aten.convolution]
    buf286 = extern_kernels.convolution(reinterpret_tensor(buf284, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf285, arg185_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf286, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg185_1
    buf287 = buf260; del buf260  # reuse
    buf288 = buf259; del buf259  # reuse
    buf290 = buf262; del buf262  # reuse
    cpp_fused_native_layer_norm_43(c_void_p(buf286.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf290.data_ptr()))
    del arg186_1
    del arg187_1
    del buf286
    buf291 = buf263; del buf263  # reuse
    # Source Nodes: [l__mod___blocks_2_2_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg189_1, reinterpret_tensor(buf290, (392, 320), (320, 1), 0), reinterpret_tensor(arg188_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf291)
    del arg188_1
    del arg189_1
    buf292 = reinterpret_tensor(buf226, (1568, 320), (320, 1), 0); del buf226  # reuse
    # Source Nodes: [l__mod___blocks_2_2_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg183_1, reinterpret_tensor(buf284, (1568, 320), (320, 1), 0), reinterpret_tensor(arg182_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf292)
    del arg182_1
    del arg183_1
    del buf284
    # Source Nodes: [x_173], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf293 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf292, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf291, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf291, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf294 = buf293[0]
    del buf293
    buf301 = buf292; del buf292  # reuse
    # Source Nodes: [x_175], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg191_1, reinterpret_tensor(buf294, (1568, 320), (320, 1), 0), reinterpret_tensor(arg190_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf301)
    del arg190_1
    del arg191_1
    buf302 = reinterpret_tensor(buf301, (8, 196, 320), (62720, 320, 1), 0); del buf301  # reuse
    buf303 = buf282; del buf282  # reuse
    buf304 = buf281; del buf281  # reuse
    buf306 = reinterpret_tensor(buf294, (8, 196, 320), (62720, 320, 1), 0); del buf294  # reuse
    cpp_fused_add_native_layer_norm_44(c_void_p(buf302.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()))
    del arg192_1
    del arg193_1
    del buf251
    buf307 = reinterpret_tensor(buf279, (1568, 1280), (1280, 1), 0); del buf279  # reuse
    # Source Nodes: [x_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg195_1, reinterpret_tensor(buf306, (1568, 320), (320, 1), 0), reinterpret_tensor(arg194_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf307)
    del arg194_1
    del arg195_1
    buf308 = reinterpret_tensor(buf307, (8, 196, 1280), (250880, 1280, 1), 0); del buf307  # reuse
    cpp_fused_gelu_45(c_void_p(buf308.data_ptr()))
    buf309 = reinterpret_tensor(buf306, (1568, 320), (320, 1), 0); del buf306  # reuse
    # Source Nodes: [x_182], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg197_1, reinterpret_tensor(buf308, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg196_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf309)
    del arg196_1
    del arg197_1
    buf310 = buf304; del buf304  # reuse
    buf311 = buf303; del buf303  # reuse
    buf313 = reinterpret_tensor(buf280, (8, 196, 320), (62720, 320, 1), 0); del buf280  # reuse
    buf314 = buf285; del buf285  # reuse
    cpp_fused_add_convolution_native_layer_norm_46(c_void_p(buf302.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    del arg198_1
    del arg199_1
    del arg202_1
    # Source Nodes: [l__mod___blocks_2_3_attn_sr], Original ATen: [aten.convolution]
    buf315 = extern_kernels.convolution(reinterpret_tensor(buf313, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf314, arg203_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf315, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg203_1
    buf316 = buf288; del buf288  # reuse
    buf317 = buf287; del buf287  # reuse
    buf319 = buf290; del buf290  # reuse
    cpp_fused_native_layer_norm_47(c_void_p(buf315.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()))
    del arg204_1
    del arg205_1
    del buf315
    buf320 = buf291; del buf291  # reuse
    # Source Nodes: [l__mod___blocks_2_3_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg207_1, reinterpret_tensor(buf319, (392, 320), (320, 1), 0), reinterpret_tensor(arg206_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf320)
    del arg206_1
    del arg207_1
    buf321 = buf273; del buf273  # reuse
    # Source Nodes: [l__mod___blocks_2_3_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg201_1, reinterpret_tensor(buf313, (1568, 320), (320, 1), 0), reinterpret_tensor(arg200_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf321)
    del arg200_1
    del arg201_1
    # Source Nodes: [x_189], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf322 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf321, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf320, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf320, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf323 = buf322[0]
    del buf322
    buf330 = buf321; del buf321  # reuse
    # Source Nodes: [x_191], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg209_1, reinterpret_tensor(buf323, (1568, 320), (320, 1), 0), reinterpret_tensor(arg208_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf330)
    del arg208_1
    del arg209_1
    buf331 = buf311; del buf311  # reuse
    buf332 = buf310; del buf310  # reuse
    buf334 = reinterpret_tensor(buf323, (8, 196, 320), (62720, 320, 1), 0); del buf323  # reuse
    cpp_fused_add_native_layer_norm_48(c_void_p(buf302.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()))
    del arg210_1
    del arg211_1
    buf335 = reinterpret_tensor(buf308, (1568, 1280), (1280, 1), 0); del buf308  # reuse
    # Source Nodes: [x_194], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg213_1, reinterpret_tensor(buf334, (1568, 320), (320, 1), 0), reinterpret_tensor(arg212_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf335)
    del arg212_1
    del arg213_1
    buf336 = reinterpret_tensor(buf335, (8, 196, 1280), (250880, 1280, 1), 0); del buf335  # reuse
    cpp_fused_gelu_49(c_void_p(buf336.data_ptr()))
    buf337 = reinterpret_tensor(buf334, (1568, 320), (320, 1), 0); del buf334  # reuse
    # Source Nodes: [x_198], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg215_1, reinterpret_tensor(buf336, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg214_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf337)
    del arg214_1
    del arg215_1
    buf338 = buf332; del buf332  # reuse
    buf339 = buf331; del buf331  # reuse
    buf341 = buf313; del buf313  # reuse
    buf342 = buf314; del buf314  # reuse
    cpp_fused_add_convolution_native_layer_norm_50(c_void_p(buf302.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    del arg216_1
    del arg217_1
    del arg220_1
    # Source Nodes: [l__mod___blocks_2_4_attn_sr], Original ATen: [aten.convolution]
    buf343 = extern_kernels.convolution(reinterpret_tensor(buf341, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf342, arg221_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf343, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg221_1
    buf344 = buf317; del buf317  # reuse
    buf345 = buf316; del buf316  # reuse
    buf347 = buf319; del buf319  # reuse
    cpp_fused_native_layer_norm_51(c_void_p(buf343.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()))
    del arg222_1
    del arg223_1
    del buf343
    buf348 = buf320; del buf320  # reuse
    # Source Nodes: [l__mod___blocks_2_4_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg225_1, reinterpret_tensor(buf347, (392, 320), (320, 1), 0), reinterpret_tensor(arg224_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf348)
    del arg224_1
    del arg225_1
    buf349 = reinterpret_tensor(buf252, (1568, 320), (320, 1), 0); del buf252  # reuse
    # Source Nodes: [l__mod___blocks_2_4_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg219_1, reinterpret_tensor(buf341, (1568, 320), (320, 1), 0), reinterpret_tensor(arg218_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf349)
    del arg218_1
    del arg219_1
    del buf341
    # Source Nodes: [x_205], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf350 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf349, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf348, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf348, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf351 = buf350[0]
    del buf350
    buf358 = buf349; del buf349  # reuse
    # Source Nodes: [x_207], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg227_1, reinterpret_tensor(buf351, (1568, 320), (320, 1), 0), reinterpret_tensor(arg226_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf358)
    del arg226_1
    del arg227_1
    buf359 = reinterpret_tensor(buf358, (8, 196, 320), (62720, 320, 1), 0); del buf358  # reuse
    buf360 = buf339; del buf339  # reuse
    buf361 = buf338; del buf338  # reuse
    buf363 = reinterpret_tensor(buf351, (8, 196, 320), (62720, 320, 1), 0); del buf351  # reuse
    cpp_fused_add_native_layer_norm_52(c_void_p(buf359.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf363.data_ptr()))
    del arg228_1
    del arg229_1
    del buf302
    buf364 = reinterpret_tensor(buf336, (1568, 1280), (1280, 1), 0); del buf336  # reuse
    # Source Nodes: [x_210], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg231_1, reinterpret_tensor(buf363, (1568, 320), (320, 1), 0), reinterpret_tensor(arg230_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf364)
    del arg230_1
    del arg231_1
    buf365 = reinterpret_tensor(buf364, (8, 196, 1280), (250880, 1280, 1), 0); del buf364  # reuse
    cpp_fused_gelu_53(c_void_p(buf365.data_ptr()))
    buf366 = reinterpret_tensor(buf363, (1568, 320), (320, 1), 0); del buf363  # reuse
    # Source Nodes: [x_214], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg233_1, reinterpret_tensor(buf365, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg232_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf366)
    del arg232_1
    del arg233_1
    buf367 = buf361; del buf361  # reuse
    buf368 = buf360; del buf360  # reuse
    buf370 = reinterpret_tensor(buf337, (8, 196, 320), (62720, 320, 1), 0); del buf337  # reuse
    buf371 = buf342; del buf342  # reuse
    cpp_fused_add_convolution_native_layer_norm_54(c_void_p(buf359.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    del arg234_1
    del arg235_1
    del arg238_1
    # Source Nodes: [l__mod___blocks_2_5_attn_sr], Original ATen: [aten.convolution]
    buf372 = extern_kernels.convolution(reinterpret_tensor(buf370, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf371, arg239_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf372, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg239_1
    buf373 = buf345; del buf345  # reuse
    buf374 = buf344; del buf344  # reuse
    buf376 = buf347; del buf347  # reuse
    cpp_fused_native_layer_norm_55(c_void_p(buf372.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()))
    del arg240_1
    del arg241_1
    del buf372
    buf377 = buf348; del buf348  # reuse
    # Source Nodes: [l__mod___blocks_2_5_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg243_1, reinterpret_tensor(buf376, (392, 320), (320, 1), 0), reinterpret_tensor(arg242_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf377)
    del arg242_1
    del arg243_1
    buf378 = buf330; del buf330  # reuse
    # Source Nodes: [l__mod___blocks_2_5_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg237_1, reinterpret_tensor(buf370, (1568, 320), (320, 1), 0), reinterpret_tensor(arg236_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf378)
    del arg236_1
    del arg237_1
    # Source Nodes: [x_221], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf379 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf378, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf377, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf377, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf380 = buf379[0]
    del buf379
    buf387 = buf378; del buf378  # reuse
    # Source Nodes: [x_223], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg245_1, reinterpret_tensor(buf380, (1568, 320), (320, 1), 0), reinterpret_tensor(arg244_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf387)
    del arg244_1
    del arg245_1
    buf388 = buf368; del buf368  # reuse
    buf389 = buf367; del buf367  # reuse
    buf391 = reinterpret_tensor(buf380, (8, 196, 320), (62720, 320, 1), 0); del buf380  # reuse
    cpp_fused_add_native_layer_norm_56(c_void_p(buf359.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf391.data_ptr()))
    del arg246_1
    del arg247_1
    buf392 = reinterpret_tensor(buf365, (1568, 1280), (1280, 1), 0); del buf365  # reuse
    # Source Nodes: [x_226], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg249_1, reinterpret_tensor(buf391, (1568, 320), (320, 1), 0), reinterpret_tensor(arg248_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf392)
    del arg248_1
    del arg249_1
    buf393 = reinterpret_tensor(buf392, (8, 196, 1280), (250880, 1280, 1), 0); del buf392  # reuse
    cpp_fused_gelu_57(c_void_p(buf393.data_ptr()))
    buf394 = reinterpret_tensor(buf391, (1568, 320), (320, 1), 0); del buf391  # reuse
    # Source Nodes: [x_230], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg251_1, reinterpret_tensor(buf393, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg250_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf394)
    del arg250_1
    del arg251_1
    buf395 = buf389; del buf389  # reuse
    buf396 = buf388; del buf388  # reuse
    buf398 = buf370; del buf370  # reuse
    buf399 = buf371; del buf371  # reuse
    cpp_fused_add_convolution_native_layer_norm_58(c_void_p(buf359.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()))
    del arg252_1
    del arg253_1
    del arg256_1
    # Source Nodes: [l__mod___blocks_2_6_attn_sr], Original ATen: [aten.convolution]
    buf400 = extern_kernels.convolution(reinterpret_tensor(buf398, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf399, arg257_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf400, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg257_1
    buf401 = buf374; del buf374  # reuse
    buf402 = buf373; del buf373  # reuse
    buf404 = buf376; del buf376  # reuse
    cpp_fused_native_layer_norm_59(c_void_p(buf400.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf404.data_ptr()))
    del arg258_1
    del arg259_1
    del buf400
    buf405 = buf377; del buf377  # reuse
    # Source Nodes: [l__mod___blocks_2_6_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg261_1, reinterpret_tensor(buf404, (392, 320), (320, 1), 0), reinterpret_tensor(arg260_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf405)
    del arg260_1
    del arg261_1
    buf406 = buf309; del buf309  # reuse
    # Source Nodes: [l__mod___blocks_2_6_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg255_1, reinterpret_tensor(buf398, (1568, 320), (320, 1), 0), reinterpret_tensor(arg254_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf406)
    del arg254_1
    del arg255_1
    del buf398
    # Source Nodes: [x_237], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf407 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf406, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf405, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf405, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf408 = buf407[0]
    del buf407
    buf415 = buf406; del buf406  # reuse
    # Source Nodes: [x_239], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg263_1, reinterpret_tensor(buf408, (1568, 320), (320, 1), 0), reinterpret_tensor(arg262_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf415)
    del arg262_1
    del arg263_1
    buf416 = reinterpret_tensor(buf415, (8, 196, 320), (62720, 320, 1), 0); del buf415  # reuse
    buf417 = buf396; del buf396  # reuse
    buf418 = buf395; del buf395  # reuse
    buf420 = reinterpret_tensor(buf408, (8, 196, 320), (62720, 320, 1), 0); del buf408  # reuse
    cpp_fused_add_native_layer_norm_60(c_void_p(buf416.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf420.data_ptr()))
    del arg264_1
    del arg265_1
    del buf359
    buf421 = reinterpret_tensor(buf393, (1568, 1280), (1280, 1), 0); del buf393  # reuse
    # Source Nodes: [x_242], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg267_1, reinterpret_tensor(buf420, (1568, 320), (320, 1), 0), reinterpret_tensor(arg266_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf421)
    del arg266_1
    del arg267_1
    buf422 = reinterpret_tensor(buf421, (8, 196, 1280), (250880, 1280, 1), 0); del buf421  # reuse
    cpp_fused_gelu_61(c_void_p(buf422.data_ptr()))
    buf423 = reinterpret_tensor(buf420, (1568, 320), (320, 1), 0); del buf420  # reuse
    # Source Nodes: [x_246], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg269_1, reinterpret_tensor(buf422, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg268_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf423)
    del arg268_1
    del arg269_1
    buf424 = buf418; del buf418  # reuse
    buf425 = buf417; del buf417  # reuse
    buf427 = reinterpret_tensor(buf394, (8, 196, 320), (62720, 320, 1), 0); del buf394  # reuse
    buf428 = buf399; del buf399  # reuse
    cpp_fused_add_convolution_native_layer_norm_62(c_void_p(buf416.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()))
    del arg270_1
    del arg271_1
    del arg274_1
    # Source Nodes: [l__mod___blocks_2_7_attn_sr], Original ATen: [aten.convolution]
    buf429 = extern_kernels.convolution(reinterpret_tensor(buf427, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf428, arg275_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf429, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg275_1
    buf430 = buf402; del buf402  # reuse
    buf431 = buf401; del buf401  # reuse
    buf433 = buf404; del buf404  # reuse
    cpp_fused_native_layer_norm_63(c_void_p(buf429.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf433.data_ptr()))
    del arg276_1
    del arg277_1
    del buf429
    buf434 = buf405; del buf405  # reuse
    # Source Nodes: [l__mod___blocks_2_7_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg279_1, reinterpret_tensor(buf433, (392, 320), (320, 1), 0), reinterpret_tensor(arg278_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf434)
    del arg278_1
    del arg279_1
    buf435 = buf387; del buf387  # reuse
    # Source Nodes: [l__mod___blocks_2_7_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg273_1, reinterpret_tensor(buf427, (1568, 320), (320, 1), 0), reinterpret_tensor(arg272_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf435)
    del arg272_1
    del arg273_1
    # Source Nodes: [x_253], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf436 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf435, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf434, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf434, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf437 = buf436[0]
    del buf436
    buf444 = buf435; del buf435  # reuse
    # Source Nodes: [x_255], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg281_1, reinterpret_tensor(buf437, (1568, 320), (320, 1), 0), reinterpret_tensor(arg280_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf444)
    del arg280_1
    del arg281_1
    buf445 = buf425; del buf425  # reuse
    buf446 = buf424; del buf424  # reuse
    buf448 = reinterpret_tensor(buf437, (8, 196, 320), (62720, 320, 1), 0); del buf437  # reuse
    cpp_fused_add_native_layer_norm_64(c_void_p(buf416.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf448.data_ptr()))
    del arg282_1
    del arg283_1
    buf449 = reinterpret_tensor(buf422, (1568, 1280), (1280, 1), 0); del buf422  # reuse
    # Source Nodes: [x_258], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg285_1, reinterpret_tensor(buf448, (1568, 320), (320, 1), 0), reinterpret_tensor(arg284_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf449)
    del arg284_1
    del arg285_1
    buf450 = reinterpret_tensor(buf449, (8, 196, 1280), (250880, 1280, 1), 0); del buf449  # reuse
    cpp_fused_gelu_65(c_void_p(buf450.data_ptr()))
    buf451 = reinterpret_tensor(buf448, (1568, 320), (320, 1), 0); del buf448  # reuse
    # Source Nodes: [x_262], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg287_1, reinterpret_tensor(buf450, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg286_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf451)
    del arg286_1
    del arg287_1
    buf452 = buf446; del buf446  # reuse
    buf453 = buf445; del buf445  # reuse
    buf455 = buf427; del buf427  # reuse
    buf456 = buf428; del buf428  # reuse
    cpp_fused_add_convolution_native_layer_norm_66(c_void_p(buf416.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()))
    del arg288_1
    del arg289_1
    del arg292_1
    # Source Nodes: [l__mod___blocks_2_8_attn_sr], Original ATen: [aten.convolution]
    buf457 = extern_kernels.convolution(reinterpret_tensor(buf455, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf456, arg293_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf457, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg293_1
    buf458 = buf431; del buf431  # reuse
    buf459 = buf430; del buf430  # reuse
    buf461 = buf433; del buf433  # reuse
    cpp_fused_native_layer_norm_67(c_void_p(buf457.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf461.data_ptr()))
    del arg294_1
    del arg295_1
    del buf457
    buf462 = buf434; del buf434  # reuse
    # Source Nodes: [l__mod___blocks_2_8_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg297_1, reinterpret_tensor(buf461, (392, 320), (320, 1), 0), reinterpret_tensor(arg296_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf462)
    del arg296_1
    del arg297_1
    buf463 = buf366; del buf366  # reuse
    # Source Nodes: [l__mod___blocks_2_8_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg291_1, reinterpret_tensor(buf455, (1568, 320), (320, 1), 0), reinterpret_tensor(arg290_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf463)
    del arg290_1
    del arg291_1
    del buf455
    # Source Nodes: [x_269], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf464 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf463, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf462, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf462, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf465 = buf464[0]
    del buf464
    buf472 = buf463; del buf463  # reuse
    # Source Nodes: [x_271], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg299_1, reinterpret_tensor(buf465, (1568, 320), (320, 1), 0), reinterpret_tensor(arg298_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf472)
    del arg298_1
    del arg299_1
    buf473 = reinterpret_tensor(buf472, (8, 196, 320), (62720, 320, 1), 0); del buf472  # reuse
    buf474 = buf453; del buf453  # reuse
    buf475 = buf452; del buf452  # reuse
    buf477 = reinterpret_tensor(buf465, (8, 196, 320), (62720, 320, 1), 0); del buf465  # reuse
    cpp_fused_add_native_layer_norm_68(c_void_p(buf473.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf477.data_ptr()))
    del arg300_1
    del arg301_1
    del buf416
    buf478 = reinterpret_tensor(buf450, (1568, 1280), (1280, 1), 0); del buf450  # reuse
    # Source Nodes: [x_274], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg303_1, reinterpret_tensor(buf477, (1568, 320), (320, 1), 0), reinterpret_tensor(arg302_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf478)
    del arg302_1
    del arg303_1
    buf479 = reinterpret_tensor(buf478, (8, 196, 1280), (250880, 1280, 1), 0); del buf478  # reuse
    cpp_fused_gelu_69(c_void_p(buf479.data_ptr()))
    buf480 = reinterpret_tensor(buf477, (1568, 320), (320, 1), 0); del buf477  # reuse
    # Source Nodes: [x_278], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg305_1, reinterpret_tensor(buf479, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg304_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf480)
    del arg304_1
    del arg305_1
    buf481 = buf475; del buf475  # reuse
    buf482 = buf474; del buf474  # reuse
    buf484 = reinterpret_tensor(buf451, (8, 196, 320), (62720, 320, 1), 0); del buf451  # reuse
    buf485 = buf456; del buf456  # reuse
    cpp_fused_add_convolution_native_layer_norm_70(c_void_p(buf473.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf485.data_ptr()))
    del arg306_1
    del arg307_1
    del arg310_1
    # Source Nodes: [l__mod___blocks_2_9_attn_sr], Original ATen: [aten.convolution]
    buf486 = extern_kernels.convolution(reinterpret_tensor(buf484, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf485, arg311_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf486, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg311_1
    buf487 = buf459; del buf459  # reuse
    buf488 = buf458; del buf458  # reuse
    buf490 = buf461; del buf461  # reuse
    cpp_fused_native_layer_norm_71(c_void_p(buf486.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf490.data_ptr()))
    del arg312_1
    del arg313_1
    del buf486
    buf491 = buf462; del buf462  # reuse
    # Source Nodes: [l__mod___blocks_2_9_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg315_1, reinterpret_tensor(buf490, (392, 320), (320, 1), 0), reinterpret_tensor(arg314_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf491)
    del arg314_1
    del arg315_1
    buf492 = buf444; del buf444  # reuse
    # Source Nodes: [l__mod___blocks_2_9_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg309_1, reinterpret_tensor(buf484, (1568, 320), (320, 1), 0), reinterpret_tensor(arg308_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf492)
    del arg308_1
    del arg309_1
    # Source Nodes: [x_285], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf493 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf492, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf491, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf491, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf494 = buf493[0]
    del buf493
    buf501 = buf492; del buf492  # reuse
    # Source Nodes: [x_287], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg317_1, reinterpret_tensor(buf494, (1568, 320), (320, 1), 0), reinterpret_tensor(arg316_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf501)
    del arg316_1
    del arg317_1
    buf502 = buf482; del buf482  # reuse
    buf503 = buf481; del buf481  # reuse
    buf505 = reinterpret_tensor(buf494, (8, 196, 320), (62720, 320, 1), 0); del buf494  # reuse
    cpp_fused_add_native_layer_norm_72(c_void_p(buf473.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf505.data_ptr()))
    del arg318_1
    del arg319_1
    buf506 = reinterpret_tensor(buf479, (1568, 1280), (1280, 1), 0); del buf479  # reuse
    # Source Nodes: [x_290], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg321_1, reinterpret_tensor(buf505, (1568, 320), (320, 1), 0), reinterpret_tensor(arg320_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf506)
    del arg320_1
    del arg321_1
    buf507 = reinterpret_tensor(buf506, (8, 196, 1280), (250880, 1280, 1), 0); del buf506  # reuse
    cpp_fused_gelu_73(c_void_p(buf507.data_ptr()))
    buf508 = reinterpret_tensor(buf505, (1568, 320), (320, 1), 0); del buf505  # reuse
    # Source Nodes: [x_294], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg323_1, reinterpret_tensor(buf507, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg322_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf508)
    del arg322_1
    del arg323_1
    buf509 = buf503; del buf503  # reuse
    buf510 = buf502; del buf502  # reuse
    buf512 = buf484; del buf484  # reuse
    buf513 = buf485; del buf485  # reuse
    cpp_fused_add_convolution_native_layer_norm_74(c_void_p(buf473.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()))
    del arg324_1
    del arg325_1
    del arg328_1
    # Source Nodes: [l__mod___blocks_2_10_attn_sr], Original ATen: [aten.convolution]
    buf514 = extern_kernels.convolution(reinterpret_tensor(buf512, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf513, arg329_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf514, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg329_1
    buf515 = buf488; del buf488  # reuse
    buf516 = buf487; del buf487  # reuse
    buf518 = buf490; del buf490  # reuse
    cpp_fused_native_layer_norm_75(c_void_p(buf514.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf518.data_ptr()))
    del arg330_1
    del arg331_1
    del buf514
    buf519 = buf491; del buf491  # reuse
    # Source Nodes: [l__mod___blocks_2_10_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg333_1, reinterpret_tensor(buf518, (392, 320), (320, 1), 0), reinterpret_tensor(arg332_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf519)
    del arg332_1
    del arg333_1
    buf520 = buf423; del buf423  # reuse
    # Source Nodes: [l__mod___blocks_2_10_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg327_1, reinterpret_tensor(buf512, (1568, 320), (320, 1), 0), reinterpret_tensor(arg326_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf520)
    del arg326_1
    del arg327_1
    del buf512
    # Source Nodes: [x_301], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf521 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf520, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf519, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf519, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf522 = buf521[0]
    del buf521
    buf529 = buf520; del buf520  # reuse
    # Source Nodes: [x_303], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg335_1, reinterpret_tensor(buf522, (1568, 320), (320, 1), 0), reinterpret_tensor(arg334_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf529)
    del arg334_1
    del arg335_1
    buf530 = reinterpret_tensor(buf529, (8, 196, 320), (62720, 320, 1), 0); del buf529  # reuse
    buf531 = buf510; del buf510  # reuse
    buf532 = buf509; del buf509  # reuse
    buf534 = reinterpret_tensor(buf522, (8, 196, 320), (62720, 320, 1), 0); del buf522  # reuse
    cpp_fused_add_native_layer_norm_76(c_void_p(buf530.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf534.data_ptr()))
    del arg336_1
    del arg337_1
    del buf473
    buf535 = reinterpret_tensor(buf507, (1568, 1280), (1280, 1), 0); del buf507  # reuse
    # Source Nodes: [x_306], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg339_1, reinterpret_tensor(buf534, (1568, 320), (320, 1), 0), reinterpret_tensor(arg338_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf535)
    del arg338_1
    del arg339_1
    buf536 = reinterpret_tensor(buf535, (8, 196, 1280), (250880, 1280, 1), 0); del buf535  # reuse
    cpp_fused_gelu_77(c_void_p(buf536.data_ptr()))
    buf537 = reinterpret_tensor(buf534, (1568, 320), (320, 1), 0); del buf534  # reuse
    # Source Nodes: [x_310], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg341_1, reinterpret_tensor(buf536, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg340_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf537)
    del arg340_1
    del arg341_1
    buf538 = buf532; del buf532  # reuse
    buf539 = buf531; del buf531  # reuse
    buf541 = reinterpret_tensor(buf508, (8, 196, 320), (62720, 320, 1), 0); del buf508  # reuse
    buf542 = buf513; del buf513  # reuse
    cpp_fused_add_convolution_native_layer_norm_78(c_void_p(buf530.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()))
    del arg342_1
    del arg343_1
    del arg346_1
    # Source Nodes: [l__mod___blocks_2_11_attn_sr], Original ATen: [aten.convolution]
    buf543 = extern_kernels.convolution(reinterpret_tensor(buf541, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf542, arg347_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf543, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg347_1
    buf544 = buf516; del buf516  # reuse
    buf545 = buf515; del buf515  # reuse
    buf547 = buf518; del buf518  # reuse
    cpp_fused_native_layer_norm_79(c_void_p(buf543.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf547.data_ptr()))
    del arg348_1
    del arg349_1
    del buf543
    buf548 = buf519; del buf519  # reuse
    # Source Nodes: [l__mod___blocks_2_11_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg351_1, reinterpret_tensor(buf547, (392, 320), (320, 1), 0), reinterpret_tensor(arg350_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf548)
    del arg350_1
    del arg351_1
    buf549 = buf501; del buf501  # reuse
    # Source Nodes: [l__mod___blocks_2_11_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg345_1, reinterpret_tensor(buf541, (1568, 320), (320, 1), 0), reinterpret_tensor(arg344_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf549)
    del arg344_1
    del arg345_1
    # Source Nodes: [x_317], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf550 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf549, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf548, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf548, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf551 = buf550[0]
    del buf550
    buf558 = buf549; del buf549  # reuse
    # Source Nodes: [x_319], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg353_1, reinterpret_tensor(buf551, (1568, 320), (320, 1), 0), reinterpret_tensor(arg352_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf558)
    del arg352_1
    del arg353_1
    buf559 = buf539; del buf539  # reuse
    buf560 = buf538; del buf538  # reuse
    buf562 = reinterpret_tensor(buf551, (8, 196, 320), (62720, 320, 1), 0); del buf551  # reuse
    cpp_fused_add_native_layer_norm_80(c_void_p(buf530.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf562.data_ptr()))
    del arg354_1
    del arg355_1
    buf563 = reinterpret_tensor(buf536, (1568, 1280), (1280, 1), 0); del buf536  # reuse
    # Source Nodes: [x_322], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg357_1, reinterpret_tensor(buf562, (1568, 320), (320, 1), 0), reinterpret_tensor(arg356_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf563)
    del arg356_1
    del arg357_1
    buf564 = reinterpret_tensor(buf563, (8, 196, 1280), (250880, 1280, 1), 0); del buf563  # reuse
    cpp_fused_gelu_81(c_void_p(buf564.data_ptr()))
    buf565 = reinterpret_tensor(buf562, (1568, 320), (320, 1), 0); del buf562  # reuse
    # Source Nodes: [x_326], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg359_1, reinterpret_tensor(buf564, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg358_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf565)
    del arg358_1
    del arg359_1
    buf566 = buf560; del buf560  # reuse
    buf567 = buf559; del buf559  # reuse
    buf569 = buf541; del buf541  # reuse
    buf570 = buf542; del buf542  # reuse
    cpp_fused_add_convolution_native_layer_norm_82(c_void_p(buf530.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()))
    del arg360_1
    del arg361_1
    del arg364_1
    # Source Nodes: [l__mod___blocks_2_12_attn_sr], Original ATen: [aten.convolution]
    buf571 = extern_kernels.convolution(reinterpret_tensor(buf569, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf570, arg365_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf571, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg365_1
    buf572 = buf545; del buf545  # reuse
    buf573 = buf544; del buf544  # reuse
    buf575 = buf547; del buf547  # reuse
    cpp_fused_native_layer_norm_83(c_void_p(buf571.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf575.data_ptr()))
    del arg366_1
    del arg367_1
    del buf571
    buf576 = buf548; del buf548  # reuse
    # Source Nodes: [l__mod___blocks_2_12_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg369_1, reinterpret_tensor(buf575, (392, 320), (320, 1), 0), reinterpret_tensor(arg368_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf576)
    del arg368_1
    del arg369_1
    buf577 = buf480; del buf480  # reuse
    # Source Nodes: [l__mod___blocks_2_12_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg363_1, reinterpret_tensor(buf569, (1568, 320), (320, 1), 0), reinterpret_tensor(arg362_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf577)
    del arg362_1
    del arg363_1
    del buf569
    # Source Nodes: [x_333], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf578 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf577, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf576, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf576, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf579 = buf578[0]
    del buf578
    buf586 = buf577; del buf577  # reuse
    # Source Nodes: [x_335], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg371_1, reinterpret_tensor(buf579, (1568, 320), (320, 1), 0), reinterpret_tensor(arg370_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf586)
    del arg370_1
    del arg371_1
    buf587 = reinterpret_tensor(buf586, (8, 196, 320), (62720, 320, 1), 0); del buf586  # reuse
    buf588 = buf567; del buf567  # reuse
    buf589 = buf566; del buf566  # reuse
    buf591 = reinterpret_tensor(buf579, (8, 196, 320), (62720, 320, 1), 0); del buf579  # reuse
    cpp_fused_add_native_layer_norm_84(c_void_p(buf587.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf591.data_ptr()))
    del arg372_1
    del arg373_1
    del buf530
    buf592 = reinterpret_tensor(buf564, (1568, 1280), (1280, 1), 0); del buf564  # reuse
    # Source Nodes: [x_338], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg375_1, reinterpret_tensor(buf591, (1568, 320), (320, 1), 0), reinterpret_tensor(arg374_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf592)
    del arg374_1
    del arg375_1
    buf593 = reinterpret_tensor(buf592, (8, 196, 1280), (250880, 1280, 1), 0); del buf592  # reuse
    cpp_fused_gelu_85(c_void_p(buf593.data_ptr()))
    buf594 = reinterpret_tensor(buf591, (1568, 320), (320, 1), 0); del buf591  # reuse
    # Source Nodes: [x_342], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg377_1, reinterpret_tensor(buf593, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg376_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf594)
    del arg376_1
    del arg377_1
    buf595 = buf589; del buf589  # reuse
    buf596 = buf588; del buf588  # reuse
    buf598 = reinterpret_tensor(buf565, (8, 196, 320), (62720, 320, 1), 0); del buf565  # reuse
    buf599 = buf570; del buf570  # reuse
    cpp_fused_add_convolution_native_layer_norm_86(c_void_p(buf587.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()))
    del arg378_1
    del arg379_1
    del arg382_1
    # Source Nodes: [l__mod___blocks_2_13_attn_sr], Original ATen: [aten.convolution]
    buf600 = extern_kernels.convolution(reinterpret_tensor(buf598, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf599, arg383_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf600, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg383_1
    buf601 = buf573; del buf573  # reuse
    buf602 = buf572; del buf572  # reuse
    buf604 = buf575; del buf575  # reuse
    cpp_fused_native_layer_norm_87(c_void_p(buf600.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf604.data_ptr()))
    del arg384_1
    del arg385_1
    del buf600
    buf605 = buf576; del buf576  # reuse
    # Source Nodes: [l__mod___blocks_2_13_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg387_1, reinterpret_tensor(buf604, (392, 320), (320, 1), 0), reinterpret_tensor(arg386_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf605)
    del arg386_1
    del arg387_1
    buf606 = buf558; del buf558  # reuse
    # Source Nodes: [l__mod___blocks_2_13_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg381_1, reinterpret_tensor(buf598, (1568, 320), (320, 1), 0), reinterpret_tensor(arg380_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf606)
    del arg380_1
    del arg381_1
    # Source Nodes: [x_349], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf607 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf606, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf605, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf605, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf608 = buf607[0]
    del buf607
    buf615 = buf606; del buf606  # reuse
    # Source Nodes: [x_351], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg389_1, reinterpret_tensor(buf608, (1568, 320), (320, 1), 0), reinterpret_tensor(arg388_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf615)
    del arg388_1
    del arg389_1
    buf616 = buf596; del buf596  # reuse
    buf617 = buf595; del buf595  # reuse
    buf619 = reinterpret_tensor(buf608, (8, 196, 320), (62720, 320, 1), 0); del buf608  # reuse
    cpp_fused_add_native_layer_norm_88(c_void_p(buf587.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf619.data_ptr()))
    del arg390_1
    del arg391_1
    buf620 = reinterpret_tensor(buf593, (1568, 1280), (1280, 1), 0); del buf593  # reuse
    # Source Nodes: [x_354], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg393_1, reinterpret_tensor(buf619, (1568, 320), (320, 1), 0), reinterpret_tensor(arg392_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf620)
    del arg392_1
    del arg393_1
    buf621 = reinterpret_tensor(buf620, (8, 196, 1280), (250880, 1280, 1), 0); del buf620  # reuse
    cpp_fused_gelu_89(c_void_p(buf621.data_ptr()))
    buf622 = reinterpret_tensor(buf619, (1568, 320), (320, 1), 0); del buf619  # reuse
    # Source Nodes: [x_358], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg395_1, reinterpret_tensor(buf621, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg394_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf622)
    del arg394_1
    del arg395_1
    buf623 = buf617; del buf617  # reuse
    buf624 = buf616; del buf616  # reuse
    buf626 = buf598; del buf598  # reuse
    buf627 = buf599; del buf599  # reuse
    cpp_fused_add_convolution_native_layer_norm_90(c_void_p(buf587.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg400_1.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf627.data_ptr()))
    del arg396_1
    del arg397_1
    del arg400_1
    # Source Nodes: [l__mod___blocks_2_14_attn_sr], Original ATen: [aten.convolution]
    buf628 = extern_kernels.convolution(reinterpret_tensor(buf626, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf627, arg401_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf628, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg401_1
    buf629 = buf602; del buf602  # reuse
    buf630 = buf601; del buf601  # reuse
    buf632 = buf604; del buf604  # reuse
    cpp_fused_native_layer_norm_91(c_void_p(buf628.data_ptr()), c_void_p(arg402_1.data_ptr()), c_void_p(arg403_1.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf632.data_ptr()))
    del arg402_1
    del arg403_1
    del buf628
    buf633 = buf605; del buf605  # reuse
    # Source Nodes: [l__mod___blocks_2_14_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg405_1, reinterpret_tensor(buf632, (392, 320), (320, 1), 0), reinterpret_tensor(arg404_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf633)
    del arg404_1
    del arg405_1
    buf634 = buf537; del buf537  # reuse
    # Source Nodes: [l__mod___blocks_2_14_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg399_1, reinterpret_tensor(buf626, (1568, 320), (320, 1), 0), reinterpret_tensor(arg398_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf634)
    del arg398_1
    del arg399_1
    del buf626
    # Source Nodes: [x_365], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf635 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf634, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf633, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf633, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf636 = buf635[0]
    del buf635
    buf643 = buf634; del buf634  # reuse
    # Source Nodes: [x_367], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg407_1, reinterpret_tensor(buf636, (1568, 320), (320, 1), 0), reinterpret_tensor(arg406_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf643)
    del arg406_1
    del arg407_1
    buf644 = reinterpret_tensor(buf643, (8, 196, 320), (62720, 320, 1), 0); del buf643  # reuse
    buf645 = buf624; del buf624  # reuse
    buf646 = buf623; del buf623  # reuse
    buf648 = reinterpret_tensor(buf636, (8, 196, 320), (62720, 320, 1), 0); del buf636  # reuse
    cpp_fused_add_native_layer_norm_92(c_void_p(buf644.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(arg408_1.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf648.data_ptr()))
    del arg408_1
    del arg409_1
    del buf587
    buf649 = reinterpret_tensor(buf621, (1568, 1280), (1280, 1), 0); del buf621  # reuse
    # Source Nodes: [x_370], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg411_1, reinterpret_tensor(buf648, (1568, 320), (320, 1), 0), reinterpret_tensor(arg410_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf649)
    del arg410_1
    del arg411_1
    buf650 = reinterpret_tensor(buf649, (8, 196, 1280), (250880, 1280, 1), 0); del buf649  # reuse
    cpp_fused_gelu_93(c_void_p(buf650.data_ptr()))
    buf651 = reinterpret_tensor(buf648, (1568, 320), (320, 1), 0); del buf648  # reuse
    # Source Nodes: [x_374], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg413_1, reinterpret_tensor(buf650, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg412_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf651)
    del arg412_1
    del arg413_1
    buf652 = buf646; del buf646  # reuse
    buf653 = buf645; del buf645  # reuse
    buf655 = reinterpret_tensor(buf622, (8, 196, 320), (62720, 320, 1), 0); del buf622  # reuse
    buf656 = buf627; del buf627  # reuse
    cpp_fused_add_convolution_native_layer_norm_94(c_void_p(buf644.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(arg414_1.data_ptr()), c_void_p(arg415_1.data_ptr()), c_void_p(arg418_1.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf656.data_ptr()))
    del arg414_1
    del arg415_1
    del arg418_1
    # Source Nodes: [l__mod___blocks_2_15_attn_sr], Original ATen: [aten.convolution]
    buf657 = extern_kernels.convolution(reinterpret_tensor(buf655, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf656, arg419_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf657, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg419_1
    buf658 = buf630; del buf630  # reuse
    buf659 = buf629; del buf629  # reuse
    buf661 = buf632; del buf632  # reuse
    cpp_fused_native_layer_norm_95(c_void_p(buf657.data_ptr()), c_void_p(arg420_1.data_ptr()), c_void_p(arg421_1.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf661.data_ptr()))
    del arg420_1
    del arg421_1
    del buf657
    buf662 = buf633; del buf633  # reuse
    # Source Nodes: [l__mod___blocks_2_15_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg423_1, reinterpret_tensor(buf661, (392, 320), (320, 1), 0), reinterpret_tensor(arg422_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf662)
    del arg422_1
    del arg423_1
    buf663 = buf615; del buf615  # reuse
    # Source Nodes: [l__mod___blocks_2_15_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg417_1, reinterpret_tensor(buf655, (1568, 320), (320, 1), 0), reinterpret_tensor(arg416_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf663)
    del arg416_1
    del arg417_1
    # Source Nodes: [x_381], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf664 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf663, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf662, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf662, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf665 = buf664[0]
    del buf664
    buf672 = buf663; del buf663  # reuse
    # Source Nodes: [x_383], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg425_1, reinterpret_tensor(buf665, (1568, 320), (320, 1), 0), reinterpret_tensor(arg424_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf672)
    del arg424_1
    del arg425_1
    buf673 = buf653; del buf653  # reuse
    buf674 = buf652; del buf652  # reuse
    buf676 = reinterpret_tensor(buf665, (8, 196, 320), (62720, 320, 1), 0); del buf665  # reuse
    cpp_fused_add_native_layer_norm_96(c_void_p(buf644.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(arg426_1.data_ptr()), c_void_p(arg427_1.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf676.data_ptr()))
    del arg426_1
    del arg427_1
    buf677 = reinterpret_tensor(buf650, (1568, 1280), (1280, 1), 0); del buf650  # reuse
    # Source Nodes: [x_386], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg429_1, reinterpret_tensor(buf676, (1568, 320), (320, 1), 0), reinterpret_tensor(arg428_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf677)
    del arg428_1
    del arg429_1
    buf678 = reinterpret_tensor(buf677, (8, 196, 1280), (250880, 1280, 1), 0); del buf677  # reuse
    cpp_fused_gelu_97(c_void_p(buf678.data_ptr()))
    buf679 = reinterpret_tensor(buf676, (1568, 320), (320, 1), 0); del buf676  # reuse
    # Source Nodes: [x_390], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg431_1, reinterpret_tensor(buf678, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg430_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf679)
    del arg430_1
    del arg431_1
    buf680 = buf674; del buf674  # reuse
    buf681 = buf673; del buf673  # reuse
    buf683 = buf655; del buf655  # reuse
    buf684 = buf656; del buf656  # reuse
    cpp_fused_add_convolution_native_layer_norm_98(c_void_p(buf644.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(arg432_1.data_ptr()), c_void_p(arg433_1.data_ptr()), c_void_p(arg436_1.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf684.data_ptr()))
    del arg432_1
    del arg433_1
    del arg436_1
    # Source Nodes: [l__mod___blocks_2_16_attn_sr], Original ATen: [aten.convolution]
    buf685 = extern_kernels.convolution(reinterpret_tensor(buf683, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf684, arg437_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf685, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg437_1
    buf686 = buf659; del buf659  # reuse
    buf687 = buf658; del buf658  # reuse
    buf689 = buf661; del buf661  # reuse
    cpp_fused_native_layer_norm_99(c_void_p(buf685.data_ptr()), c_void_p(arg438_1.data_ptr()), c_void_p(arg439_1.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf689.data_ptr()))
    del arg438_1
    del arg439_1
    del buf685
    buf690 = buf662; del buf662  # reuse
    # Source Nodes: [l__mod___blocks_2_16_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg441_1, reinterpret_tensor(buf689, (392, 320), (320, 1), 0), reinterpret_tensor(arg440_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf690)
    del arg440_1
    del arg441_1
    buf691 = buf594; del buf594  # reuse
    # Source Nodes: [l__mod___blocks_2_16_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg435_1, reinterpret_tensor(buf683, (1568, 320), (320, 1), 0), reinterpret_tensor(arg434_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf691)
    del arg434_1
    del arg435_1
    del buf683
    # Source Nodes: [x_397], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf692 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf691, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf690, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf690, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    buf693 = buf692[0]
    del buf692
    buf700 = buf691; del buf691  # reuse
    # Source Nodes: [x_399], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg443_1, reinterpret_tensor(buf693, (1568, 320), (320, 1), 0), reinterpret_tensor(arg442_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf700)
    del arg442_1
    del arg443_1
    buf701 = reinterpret_tensor(buf700, (8, 196, 320), (62720, 320, 1), 0); del buf700  # reuse
    buf702 = buf681; del buf681  # reuse
    buf703 = buf680; del buf680  # reuse
    buf705 = reinterpret_tensor(buf693, (8, 196, 320), (62720, 320, 1), 0); del buf693  # reuse
    cpp_fused_add_native_layer_norm_100(c_void_p(buf701.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(arg444_1.data_ptr()), c_void_p(arg445_1.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf703.data_ptr()), c_void_p(buf705.data_ptr()))
    del arg444_1
    del arg445_1
    del buf644
    del buf651
    buf706 = reinterpret_tensor(buf678, (1568, 1280), (1280, 1), 0); del buf678  # reuse
    # Source Nodes: [x_402], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg447_1, reinterpret_tensor(buf705, (1568, 320), (320, 1), 0), reinterpret_tensor(arg446_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf706)
    del arg446_1
    del arg447_1
    buf707 = reinterpret_tensor(buf706, (8, 196, 1280), (250880, 1280, 1), 0); del buf706  # reuse
    cpp_fused_gelu_101(c_void_p(buf707.data_ptr()))
    buf708 = reinterpret_tensor(buf705, (1568, 320), (320, 1), 0); del buf705  # reuse
    # Source Nodes: [x_406], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg449_1, reinterpret_tensor(buf707, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg448_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf708)
    del arg448_1
    del arg449_1
    buf709 = buf703; del buf703  # reuse
    buf710 = buf702; del buf702  # reuse
    buf712 = reinterpret_tensor(buf679, (8, 196, 320), (62720, 320, 1), 0); del buf679  # reuse
    buf713 = buf684; del buf684  # reuse
    cpp_fused_add_convolution_native_layer_norm_102(c_void_p(buf701.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(arg450_1.data_ptr()), c_void_p(arg451_1.data_ptr()), c_void_p(arg454_1.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf713.data_ptr()))
    del arg450_1
    del arg451_1
    del arg454_1
    # Source Nodes: [l__mod___blocks_2_17_attn_sr], Original ATen: [aten.convolution]
    buf714 = extern_kernels.convolution(reinterpret_tensor(buf712, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), buf713, arg455_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf714, (8, 320, 7, 7), (15680, 1, 2240, 320))
    del arg455_1
    del buf713
    buf715 = buf687; del buf687  # reuse
    buf716 = buf686; del buf686  # reuse
    buf718 = buf689; del buf689  # reuse
    cpp_fused_native_layer_norm_103(c_void_p(buf714.data_ptr()), c_void_p(arg456_1.data_ptr()), c_void_p(arg457_1.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf718.data_ptr()))
    del arg456_1
    del arg457_1
    del buf714
    buf719 = buf690; del buf690  # reuse
    # Source Nodes: [l__mod___blocks_2_17_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg459_1, reinterpret_tensor(buf718, (392, 320), (320, 1), 0), reinterpret_tensor(arg458_1, (320, 640), (1, 320), 0), alpha=1, beta=1, out=buf719)
    del arg458_1
    del arg459_1
    del buf718
    buf720 = buf672; del buf672  # reuse
    # Source Nodes: [l__mod___blocks_2_17_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg453_1, reinterpret_tensor(buf712, (1568, 320), (320, 1), 0), reinterpret_tensor(arg452_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf720)
    del arg452_1
    del arg453_1
    del buf712
    # Source Nodes: [x_413], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf721 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf720, (8, 5, 196, 64), (62720, 64, 320, 1), 0), reinterpret_tensor(buf719, (8, 5, 49, 64), (31360, 64, 640, 1), 0), reinterpret_tensor(buf719, (8, 5, 49, 64), (31360, 64, 640, 1), 320))
    del buf719
    buf722 = buf721[0]
    del buf721
    buf729 = buf720; del buf720  # reuse
    # Source Nodes: [x_415], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg461_1, reinterpret_tensor(buf722, (1568, 320), (320, 1), 0), reinterpret_tensor(arg460_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf729)
    del arg460_1
    del arg461_1
    buf730 = buf710; del buf710  # reuse
    buf731 = buf709; del buf709  # reuse
    buf733 = reinterpret_tensor(buf722, (8, 196, 320), (62720, 320, 1), 0); del buf722  # reuse
    cpp_fused_add_native_layer_norm_104(c_void_p(buf701.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(arg462_1.data_ptr()), c_void_p(arg463_1.data_ptr()), c_void_p(buf730.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf733.data_ptr()))
    del arg462_1
    del arg463_1
    del buf730
    del buf731
    buf734 = reinterpret_tensor(buf707, (1568, 1280), (1280, 1), 0); del buf707  # reuse
    # Source Nodes: [x_418], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg465_1, reinterpret_tensor(buf733, (1568, 320), (320, 1), 0), reinterpret_tensor(arg464_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf734)
    del arg464_1
    del arg465_1
    buf735 = reinterpret_tensor(buf734, (8, 196, 1280), (250880, 1280, 1), 0); del buf734  # reuse
    cpp_fused_gelu_105(c_void_p(buf735.data_ptr()))
    buf736 = reinterpret_tensor(buf733, (1568, 320), (320, 1), 0); del buf733  # reuse
    # Source Nodes: [x_422], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg467_1, reinterpret_tensor(buf735, (1568, 1280), (1280, 1), 0), reinterpret_tensor(arg466_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf736)
    del arg466_1
    del arg467_1
    del buf735
    buf737 = reinterpret_tensor(buf736, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf736  # reuse
    buf738 = empty_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    cpp_fused_clone_convolution_106(c_void_p(buf737.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(arg468_1.data_ptr()), c_void_p(buf738.data_ptr()))
    del arg468_1
    del buf701
    del buf708
    del buf729
    # Source Nodes: [l__mod___patch_embeds_3_proj, x_426], Original ATen: [aten.clone, aten.convolution]
    buf739 = extern_kernels.convolution(buf737, buf738, arg469_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf739, (8, 512, 7, 7), (25088, 1, 3584, 512))
    del arg469_1
    del buf737
    del buf738
    buf740 = buf716; del buf716  # reuse
    buf741 = buf715; del buf715  # reuse
    buf743 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf744 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf745 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf747 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_107(c_void_p(buf739.data_ptr()), c_void_p(arg470_1.data_ptr()), c_void_p(arg471_1.data_ptr()), c_void_p(arg472_1.data_ptr()), c_void_p(arg473_1.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf743.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf747.data_ptr()))
    del arg470_1
    del arg471_1
    del arg472_1
    del arg473_1
    del buf740
    del buf741
    buf748 = empty((392, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_3_0_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg477_1, reinterpret_tensor(buf747, (392, 512), (512, 1), 0), reinterpret_tensor(arg476_1, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf748)
    del arg476_1
    del arg477_1
    buf749 = reinterpret_tensor(buf739, (392, 512), (512, 1), 0); del buf739  # reuse
    # Source Nodes: [l__mod___blocks_3_0_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg475_1, reinterpret_tensor(buf747, (392, 512), (512, 1), 0), reinterpret_tensor(arg474_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf749)
    del arg474_1
    del arg475_1
    # Source Nodes: [x_431], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf750 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf749, (8, 8, 49, 64), (25088, 64, 512, 1), 0), reinterpret_tensor(buf748, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf748, (8, 8, 49, 64), (50176, 64, 1024, 1), 512))
    buf751 = buf750[0]
    del buf750
    buf758 = buf749; del buf749  # reuse
    # Source Nodes: [x_433], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg479_1, reinterpret_tensor(buf751, (392, 512), (512, 1), 0), reinterpret_tensor(arg478_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf758)
    del arg478_1
    del arg479_1
    buf759 = buf745; del buf745  # reuse
    buf760 = buf744; del buf744  # reuse
    buf762 = reinterpret_tensor(buf751, (8, 49, 512), (25088, 512, 1), 0); del buf751  # reuse
    cpp_fused_add_native_layer_norm_108(c_void_p(buf743.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(arg480_1.data_ptr()), c_void_p(arg481_1.data_ptr()), c_void_p(buf759.data_ptr()), c_void_p(buf760.data_ptr()), c_void_p(buf762.data_ptr()))
    del arg480_1
    del arg481_1
    buf763 = reinterpret_tensor(buf216, (392, 2048), (2048, 1), 0); del buf216  # reuse
    # Source Nodes: [x_436], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg483_1, reinterpret_tensor(buf762, (392, 512), (512, 1), 0), reinterpret_tensor(arg482_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf763)
    del arg482_1
    del arg483_1
    buf764 = reinterpret_tensor(buf763, (8, 49, 2048), (100352, 2048, 1), 0); del buf763  # reuse
    cpp_fused_gelu_109(c_void_p(buf764.data_ptr()))
    buf765 = reinterpret_tensor(buf762, (392, 512), (512, 1), 0); del buf762  # reuse
    # Source Nodes: [x_440], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg485_1, reinterpret_tensor(buf764, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg484_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf765)
    del arg484_1
    del arg485_1
    buf766 = reinterpret_tensor(buf765, (8, 49, 512), (25088, 512, 1), 0); del buf765  # reuse
    cpp_fused_add_110(c_void_p(buf766.data_ptr()), c_void_p(buf743.data_ptr()), c_void_p(buf758.data_ptr()))
    # Source Nodes: [x_444], Original ATen: [aten.convolution]
    buf767 = extern_kernels.convolution(reinterpret_tensor(buf766, (8, 512, 7, 7), (25088, 1, 3584, 512), 0), arg486_1, arg487_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf767, (8, 512, 7, 7), (25088, 1, 3584, 512))
    del arg486_1
    del arg487_1
    buf768 = buf760; del buf760  # reuse
    buf769 = buf759; del buf759  # reuse
    buf771 = reinterpret_tensor(buf758, (8, 49, 512), (25088, 512, 1), 0); del buf758  # reuse
    cpp_fused_native_layer_norm_111(c_void_p(buf767.data_ptr()), c_void_p(buf766.data_ptr()), c_void_p(arg488_1.data_ptr()), c_void_p(arg489_1.data_ptr()), c_void_p(buf768.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(buf771.data_ptr()))
    del arg488_1
    del arg489_1
    buf772 = buf748; del buf748  # reuse
    # Source Nodes: [l__mod___blocks_3_1_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg493_1, reinterpret_tensor(buf771, (392, 512), (512, 1), 0), reinterpret_tensor(arg492_1, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf772)
    del arg492_1
    del arg493_1
    buf773 = reinterpret_tensor(buf743, (392, 512), (512, 1), 0); del buf743  # reuse
    # Source Nodes: [l__mod___blocks_3_1_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg491_1, reinterpret_tensor(buf771, (392, 512), (512, 1), 0), reinterpret_tensor(arg490_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf773)
    del arg490_1
    del arg491_1
    # Source Nodes: [x_448], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf774 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf773, (8, 8, 49, 64), (25088, 64, 512, 1), 0), reinterpret_tensor(buf772, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf772, (8, 8, 49, 64), (50176, 64, 1024, 1), 512))
    buf775 = buf774[0]
    del buf774
    buf782 = buf773; del buf773  # reuse
    # Source Nodes: [x_450], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg495_1, reinterpret_tensor(buf775, (392, 512), (512, 1), 0), reinterpret_tensor(arg494_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf782)
    del arg494_1
    del arg495_1
    buf783 = buf769; del buf769  # reuse
    buf784 = buf768; del buf768  # reuse
    buf786 = reinterpret_tensor(buf775, (8, 49, 512), (25088, 512, 1), 0); del buf775  # reuse
    cpp_fused_add_native_layer_norm_112(c_void_p(buf767.data_ptr()), c_void_p(buf766.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(arg496_1.data_ptr()), c_void_p(arg497_1.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf786.data_ptr()))
    del arg496_1
    del arg497_1
    buf787 = reinterpret_tensor(buf764, (392, 2048), (2048, 1), 0); del buf764  # reuse
    # Source Nodes: [x_453], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg499_1, reinterpret_tensor(buf786, (392, 512), (512, 1), 0), reinterpret_tensor(arg498_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf787)
    del arg498_1
    del arg499_1
    buf788 = reinterpret_tensor(buf787, (8, 49, 2048), (100352, 2048, 1), 0); del buf787  # reuse
    cpp_fused_gelu_113(c_void_p(buf788.data_ptr()))
    buf789 = reinterpret_tensor(buf786, (392, 512), (512, 1), 0); del buf786  # reuse
    # Source Nodes: [x_457], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg501_1, reinterpret_tensor(buf788, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg500_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf789)
    del arg500_1
    del arg501_1
    buf790 = buf784; del buf784  # reuse
    buf791 = buf783; del buf783  # reuse
    buf793 = buf771; del buf771  # reuse
    cpp_fused_add_native_layer_norm_114(c_void_p(buf767.data_ptr()), c_void_p(buf766.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(buf789.data_ptr()), c_void_p(arg502_1.data_ptr()), c_void_p(arg503_1.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf791.data_ptr()), c_void_p(buf793.data_ptr()))
    del arg502_1
    del arg503_1
    buf794 = buf772; del buf772  # reuse
    # Source Nodes: [l__mod___blocks_3_2_attn_kv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg507_1, reinterpret_tensor(buf793, (392, 512), (512, 1), 0), reinterpret_tensor(arg506_1, (512, 1024), (1, 512), 0), alpha=1, beta=1, out=buf794)
    del arg506_1
    del arg507_1
    buf795 = reinterpret_tensor(buf747, (392, 512), (512, 1), 0); del buf747  # reuse
    # Source Nodes: [l__mod___blocks_3_2_attn_q], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg505_1, reinterpret_tensor(buf793, (392, 512), (512, 1), 0), reinterpret_tensor(arg504_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf795)
    del arg504_1
    del arg505_1
    del buf793
    # Source Nodes: [x_461], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf796 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf795, (8, 8, 49, 64), (25088, 64, 512, 1), 0), reinterpret_tensor(buf794, (8, 8, 49, 64), (50176, 64, 1024, 1), 0), reinterpret_tensor(buf794, (8, 8, 49, 64), (50176, 64, 1024, 1), 512))
    del buf794
    buf797 = buf796[0]
    del buf796
    buf804 = buf795; del buf795  # reuse
    # Source Nodes: [x_463], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg509_1, reinterpret_tensor(buf797, (392, 512), (512, 1), 0), reinterpret_tensor(arg508_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf804)
    del arg508_1
    del arg509_1
    buf805 = reinterpret_tensor(buf804, (8, 49, 512), (25088, 512, 1), 0); del buf804  # reuse
    buf806 = buf791; del buf791  # reuse
    buf807 = buf790; del buf790  # reuse
    buf809 = reinterpret_tensor(buf797, (8, 49, 512), (25088, 512, 1), 0); del buf797  # reuse
    cpp_fused_add_native_layer_norm_115(c_void_p(buf805.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(buf766.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(buf789.data_ptr()), c_void_p(arg510_1.data_ptr()), c_void_p(arg511_1.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf809.data_ptr()))
    del arg510_1
    del arg511_1
    del buf766
    del buf767
    del buf782
    del buf789
    buf810 = reinterpret_tensor(buf788, (392, 2048), (2048, 1), 0); del buf788  # reuse
    # Source Nodes: [x_466], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg513_1, reinterpret_tensor(buf809, (392, 512), (512, 1), 0), reinterpret_tensor(arg512_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf810)
    del arg512_1
    del arg513_1
    buf811 = reinterpret_tensor(buf810, (8, 49, 2048), (100352, 2048, 1), 0); del buf810  # reuse
    cpp_fused_gelu_116(c_void_p(buf811.data_ptr()))
    buf812 = reinterpret_tensor(buf809, (392, 512), (512, 1), 0); del buf809  # reuse
    # Source Nodes: [x_470], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg515_1, reinterpret_tensor(buf811, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg514_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf812)
    del arg514_1
    del arg515_1
    del buf811
    buf813 = buf807; del buf807  # reuse
    buf814 = buf806; del buf806  # reuse
    buf816 = empty((8, 512), device='cpu', dtype=torch.float32)
    buf817 = buf816; del buf816  # reuse
    cpp_fused_add_mean_native_layer_norm_117(c_void_p(buf817.data_ptr()), c_void_p(buf805.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(arg516_1.data_ptr()), c_void_p(arg517_1.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf814.data_ptr()))
    del arg516_1
    del arg517_1
    del buf805
    del buf812
    del buf813
    del buf814
    buf818 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_473, x_475, x_476, x_478], Original ATen: [aten.add, aten.addmm, aten.mean, aten.native_layer_norm]
    extern_kernels.addmm(arg519_1, buf817, reinterpret_tensor(arg518_1, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf818)
    del arg518_1
    del arg519_1
    return (buf818, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((128, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((128, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((64, 64, 8, 8), (4096, 64, 8, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((128, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((128, 128, 4, 4), (2048, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((320, 128, 2, 2), (512, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg424_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg427_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg430_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg433_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg436_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg439_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg442_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg445_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg448_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg451_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg454_1 = rand_strided((320, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg457_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg458_1 = rand_strided((640, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg460_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg461_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg463_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg464_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg466_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg467_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((512, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg469_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg470_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg472_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg473_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg475_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg476_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg478_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg479_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg481_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg482_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg484_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg485_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg487_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg488_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg490_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg491_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg493_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg494_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg496_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg497_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg499_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg500_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg502_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg503_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg505_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg506_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg508_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg509_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg511_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg512_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg513_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg514_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg515_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg516_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg517_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg518_1 = rand_strided((1000, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg519_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg520_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('twins_pcpvt_base', benchmark_compiled_module)
